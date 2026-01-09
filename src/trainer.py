import gc
import json
import math
import os
import signal
import sys
import time
from typing import List
import random

import torch
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, RobertaConfig

from src import utils
from src.base_trainer import BaseTrainer
from src.entities import Dataset
from src.evaluation import FastEvaluator, SlowEvaluator
from src.model.model import Miner
from src.model.news_encoder import NewsEncoder
from src.loss import Loss
from src.reader import Reader


class CleanupHandler:
    """Global handler for cleanup on interrupts"""
    _dataloader = None
    _file_handle = None
    
    @classmethod
    def register_dataloader(cls, dataloader):
        cls._dataloader = dataloader
    
    @classmethod
    def register_file(cls, file_handle):
        cls._file_handle = file_handle
    
    @classmethod
    def cleanup(cls, signum=None, frame=None):
        """Cleanup resources on interrupt"""
        print("\n[INTERRUPT] Cleaning up resources...")
        
        # Close file handle
        if cls._file_handle:
            try:
                cls._file_handle.close()
                print("File handle closed")
            except Exception:
                pass
        
        # Cleanup dataloader
        if cls._dataloader is not None and hasattr(cls._dataloader, '_iterator'):
            try:
                if cls._dataloader._iterator is not None:
                    cls._dataloader._iterator._shutdown_workers()
                print("DataLoader workers shutdown")
            except Exception:
                pass
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("CUDA cache cleared")
        
        gc.collect()
        print("Cleanup complete")
        
        if signum is not None:
            sys.exit(1)


class Trainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self._tokenizer = AutoTokenizer.from_pretrained(args.pretrained_tokenizer)

        with open(args.user2id_path, mode='r', encoding='utf-8') as f:
            self._user2id = json.load(f)
        with open(args.category2id_path, mode='r', encoding='utf-8') as f:
            self._category2id = json.load(f)

        if 'fp16' in args:
            if args.fp16:
                self.scaler = GradScaler()
            else:
                self.scaler = None
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, CleanupHandler.cleanup)
        signal.signal(signal.SIGTERM, CleanupHandler.cleanup)

    def train(self):
        args = self.args
        self._log_arguments()
        self._logger.info(f'Model: {args.model_name}')

        # Read pretrained embedding (if any)
        if args.category_embed_path is not None:
            category_embed = utils.load_embed(args.category_embed_path)
        else:
            category_embed = None

        # Read data
        reader = Reader(tokenizer=self._tokenizer, max_title_length=args.max_title_length,
                        max_sapo_length=args.max_sapo_length, user2id=self._user2id, category2id=self._category2id,
                        max_his_click=args.his_length, npratio=args.npratio)
        train_dataset = reader.read_train_dataset(args.data_name, args.train_news_path, args.train_behaviors_path)
        train_dataset.set_mode(Dataset.TRAIN_MODE)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False,
                                      num_workers=args.dataloader_num_workers, collate_fn=self._collate_fn,
                                      drop_last=args.dataloader_drop_last, pin_memory=args.dataloader_pin_memory)
        if args.fast_eval:
            eval_dataset = reader.read_train_dataset(args.data_name, args.eval_news_path, args.eval_behaviors_path)
        else:
            eval_dataset = reader.read_eval_dataset(args.data_name, args.eval_news_path, args.eval_behaviors_path)
        self._log_dataset(train_dataset, eval_dataset)

        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps
        len_dataloader = len(train_dataloader)
        num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if args.max_steps is not None:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(args.max_steps %
                                                                                  num_update_steps_per_epoch > 0)
        else:
            max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(args.num_train_epochs)

        self._logger.info('------------- Start training -------------')
        self._logger.info(f'Num epochs: {num_train_epochs}')
        self._logger.info(f'Updates per epoch: {num_update_steps_per_epoch}')
        self._logger.info(f'Updates total: {max_steps}')
        self._logger.info(f'Total train batch size: {total_train_batch_size}')
        self._logger.info(f'Gradient accumulation steps: {args.gradient_accumulation_steps}')

        # Create model
        config = RobertaConfig.from_pretrained(args.pretrained_embedding)
        news_encoder = NewsEncoder.from_pretrained(args.pretrained_embedding, config=config,
                                                   apply_reduce_dim=args.apply_reduce_dim, use_sapo=args.use_sapo,
                                                   dropout=args.dropout, freeze_transformer=args.freeze_transformer,
                                                   word_embed_dim=args.word_embed_dim, combine_type=args.combine_type,
                                                   lstm_num_layers=args.lstm_num_layers, lstm_dropout=args.lstm_dropout)
        model = Miner(news_encoder=news_encoder, use_category_bias=args.use_category_bias,
                      num_context_codes=args.num_context_codes, context_code_dim=args.context_code_dim,
                      score_type=args.score_type, dropout=args.dropout, num_category=len(self._category2id),
                      category_embed_dim=args.category_embed_dim, category_pad_token_id=self._category2id['pad'],
                      category_embed=category_embed)
        model.to(self._device)
        
        # Enable multi-GPU training if requested
        if hasattr(args, 'use_multi_gpu') and args.use_multi_gpu and torch.cuda.device_count() > 1:
            self._logger.info(f'Using {torch.cuda.device_count()} GPUs with DataParallel')
            model = nn.DataParallel(model)
            self._is_multi_gpu = True
        else:
            self._is_multi_gpu = False
            
        model.zero_grad(set_to_none=True)

        # Create optimizer and scheduler
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.learning_rate, weight_decay=args.weight_decay)
        
        # Select scheduler based on args.lr_scheduler_type
        num_warmup_steps = self._get_warmup_steps(max_steps)
        if hasattr(args, 'lr_scheduler_type') and args.lr_scheduler_type == 'cosine':
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=max_steps
            )
        else:
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=max_steps
            )
        loss_calculator = self._create_loss()

        best_valid_loss = float('inf')
        best_auc_score = 0.0
        global_step = 0
        global_iteration = 0
        logging_loss = 0.0
        start_epoch = 0

        # Load checkpoint if resume_from_checkpoint is specified
        if args.resume_from_checkpoint is not None:
            self._logger.info(f'Loading checkpoint from {args.resume_from_checkpoint}')
            checkpoint = torch.load(args.resume_from_checkpoint, map_location=self._device)
            model.load_state_dict(checkpoint['model'].state_dict())
            
            # Check if we want to resume from exact step (load optimizer & scheduler)
            if hasattr(args, 'resume_training') and args.resume_training:
                # Resume training from checkpoint - load optimizer, scheduler, and step
                optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
                scheduler.load_state_dict(checkpoint['scheduler'])
                if 'global_step' in checkpoint:
                    global_step = checkpoint['global_step']
                    global_iteration = global_step * args.gradient_accumulation_steps
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch']
                self._logger.info(f'Resumed training from step {global_step}, epoch {start_epoch}')
            else:
                # Fine-tune mode - only load weights, reset optimizer & scheduler
                self._logger.info('Checkpoint loaded successfully (weights only, optimizer & scheduler reset)')

        for epoch in range(start_epoch, args.num_train_epochs):
            epoch_start_time = time.time()
            eval_time = 0.0
            self._logger.info(f'--------------- EPOCH {epoch} ---------------')
            steps_in_epoch = len(train_dataloader)
            epoch_loss = 0.0
            accumulation_factor = (self.args.gradient_accumulation_steps
                                   if steps_in_epoch > self.args.gradient_accumulation_steps else steps_in_epoch)

            for step, batch in tqdm(enumerate(train_dataloader), total=steps_in_epoch, desc=f'Train epoch {epoch}'):
                batch_loss = self._train_step(batch, model, loss_calculator, accumulation_factor)
                logging_loss += batch_loss
                epoch_loss += batch_loss
                self._log_train_step(scheduler, batch_loss, global_iteration)
                if not (not ((global_iteration + 1) % args.gradient_accumulation_steps == 0) and not (
                        args.gradient_accumulation_steps >= steps_in_epoch == (step + 1))):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        if args.fp16:
                            self.scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    # Optimizer step
                    optimizer_was_run = True
                    if args.fp16:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        optimizer.step()

                    if optimizer_was_run:
                        scheduler.step()

                    model.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % args.logging_steps == 0:
                        logging_loss /= (args.logging_steps * accumulation_factor)
                        self._log_train(scheduler, logging_loss, global_step, epoch)
                        logging_loss = 0.0

                    # Save checkpoint periodically
                    if args.save_steps is not None and global_step % args.save_steps == 0:
                        self._logger.info(f'Saving checkpoint at step {global_step}')
                        self._save_model(model, optimizer, scheduler, flag=f'checkpoint-{global_step}', 
                                       global_step=global_step, epoch=epoch)

                    if global_step % args.eval_steps == 0:
                        eval_start_time = time.time()
                        valid_loss, scores = self._eval(model, eval_dataset, loss_calculator, metrics=args.metrics)
                        self._log_eval(global_step, valid_loss, scores)

                        if 'loss' in self.args.evaluation_info and valid_loss < best_valid_loss:
                            self._logger.info(f'Best loss updates from {best_valid_loss} to {valid_loss}, '
                                              f'at global step {global_step}')
                            best_valid_loss = valid_loss
                            self._save_model(model, optimizer, scheduler, flag='bestLossModel',
                                           global_step=global_step, epoch=epoch)
                        if 'metrics' in self.args.evaluation_info and scores['auc'] > best_auc_score:
                            self._logger.info(f'Best AUC score updates from {best_auc_score} to {scores["auc"]}, '
                                              f'at global step {global_step}')
                            best_auc_score = scores['auc']
                            self._save_model(model, optimizer, scheduler, flag='bestAucModel',
                                           global_step=global_step, epoch=epoch)
                        eval_time += time.time() - eval_start_time
                global_iteration += 1

            # Evaluation at the end of each epoch
            eval_start_time = time.time()
            valid_loss, scores = self._eval(model, eval_dataset, loss_calculator, metrics=args.metrics)
            train_loss = epoch_loss / steps_in_epoch
            self._log_epoch(train_loss, valid_loss, scores, epoch)
            if 'loss' in self.args.evaluation_info and valid_loss < best_valid_loss:
                self._logger.info(f'Best loss updates from {best_valid_loss} to {valid_loss}, at epoch {epoch}')
                best_valid_loss = valid_loss
                self._save_model(model, optimizer, scheduler, flag='bestLossModel',
                               global_step=global_step, epoch=epoch)
            if 'metrics' in self.args.evaluation_info and scores['auc'] > best_auc_score:
                self._logger.info(f'Best AUC score updates from {best_auc_score} to {scores["auc"]}, at epoch {epoch}')
                best_auc_score = scores['auc']
                self._save_model(model, optimizer, scheduler, flag='bestAucModel',
                               global_step=global_step, epoch=epoch)
            eval_time += time.time() - eval_start_time

            # Log running time
            epoch_end_time = time.time()
            self._logger.info(f'Total running time of epoch: {round(epoch_end_time - epoch_start_time, ndigits=4)} (s)')
            self._logger.info(f'Total training time of epoch: '
                              f'{round(epoch_end_time - epoch_start_time - eval_time, ndigits=4)} (s)')

        # Save final model
        self._save_model(model, optimizer, scheduler, flag='finalModel',
                       global_step=global_step, epoch=num_train_epochs-1)
        self._logger.info('---  Finish training!!!  ---')

    def eval(self):
        args = self.args
        self._log_arguments()

        # Load model
        checkpoint : nn.Module = self._load_model(args.saved_model_path)
        if 'is_old_format' in checkpoint:
            model = checkpoint['model']
            if 'epoch' in checkpoint:
                self._logger.info(f'Model was trained for {checkpoint.get("epoch", "unknown")} epochs')
        else:
            self._logger.info('Rebuilding model from checkpoint...')
            category_embed = None
            model = self._build_model(category_embed)
            # Load with strict=False to ignore non-parameter buffers like position_ids
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if 'epoch' in checkpoint:
                self._logger.info(f'Model was trained for {checkpoint["epoch"]} epochs')
        model.to(self._device)
        
        # Enable multi-GPU evaluation if requested
        if hasattr(args, 'use_multi_gpu') and args.use_multi_gpu and torch.cuda.device_count() > 1:
            self._logger.info(f'Using {torch.cuda.device_count()} GPUs for evaluation with DataParallel')
            model = nn.DataParallel(model)
        
        model.eval()

        # Read eval dataset
        reader = Reader(tokenizer=self._tokenizer, max_title_length=args.max_title_length,
                        max_sapo_length=args.max_sapo_length, user2id=self._user2id, category2id=self._category2id,
                        max_his_click=args.his_length, npratio=None)
        dataset = reader.read_eval_dataset(args.data_name, args.eval_news_path, args.eval_behaviors_path)
        self._logger.info(f'Model: {self.args.model_name}')
        self._logger.info(f'Dataset: {self.args.data_name}')
        self._logger.info(f'Test dataset: {len(dataset)} samples')

        # Evaluation
        loss_calculator = self._create_loss()
        self._logger.info('----------------  Evaluation phrase  ----------------')
        loss, scores = self._eval(model, dataset, loss_calculator, metrics=args.metrics,
                                  save_result=True)
        if 'loss' in self.args.evaluation_info:
            self._logger.info('Loss {}'.format(loss))
        for metric in args.metrics:
            self._logger.info(f'Metric {metric}: {scores[metric]}')
    
    def submission_generator(self):
        args = self.args
        self._log_arguments()

        # Load model
        checkpoint : nn.Module = self._load_model(args.saved_model_path)
        if 'is_old_format' in checkpoint:
            model = checkpoint['model']
            if 'epoch' in checkpoint:
                self._logger.info(f'Model was trained for {checkpoint.get("epoch", "unknown")} epochs')
        else:
            self._logger.info('Rebuilding model from checkpoint...')
            category_embed = None
            model = self._build_model(category_embed)
            # Load with strict=False to ignore non-parameter buffers like position_ids
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if 'epoch' in checkpoint:
                self._logger.info(f'Model was trained for {checkpoint["epoch"]} epochs')
        model.to(self._device)
        
        # Enable multi-GPU evaluation if requested
        if hasattr(args, 'use_multi_gpu') and args.use_multi_gpu and torch.cuda.device_count() > 1:
            self._logger.info(f'Using {torch.cuda.device_count()} GPUs for submission generation with DataParallel')
            model = nn.DataParallel(model)
        
        model.eval()

        # Read eval dataset
        reader = Reader(tokenizer=self._tokenizer, max_title_length=args.max_title_length,
                        max_sapo_length=args.max_sapo_length, user2id=self._user2id, category2id=self._category2id,
                        max_his_click=args.his_length, npratio=None)
        dataset = reader.read_submission_dataset(args.data_name, args.eval_news_path, args.eval_behaviors_path)
        self._logger.info(f'Model: {self.args.model_name}')
        self._logger.info(f'Dataset: {self.args.data_name}')
        self._logger.info(f'Test dataset: {len(dataset)} samples')

        # Evaluation
        self._logger.info('----------------  Evaluation phrase  ----------------')
        self._submiss(model, dataset, save_result=True)
        
    def _submiss(self, model : nn.Module, dataset : Dataset, save_result: bool = False):
        model.eval()
        
        # Pre-encode ALL unique news FIRST (before sampling) to ensure all news are cached
        self._logger.info('Pre-encoding all unique news for evaluation...')
        news_embeddings_cache, encoded_news_ids = self._encode_all_news(model, dataset)
        self._logger.info(f'Cached {len(encoded_news_ids)} news embeddings')
        
        original_samples = None
        if save_result:  # Only sample during training eval, not final eval
            impression_ids = list(set([sample.impression.impression_id for sample in dataset.samples]))
            random.seed(42)  # Reproducible sampling
            sampled_impression_ids = set(random.sample(impression_ids, k=max(1, len(impression_ids))))
            # Temporarily replace _samples with filtered version
            original_samples = dataset._samples
            dataset._samples = {k: v for k, v in original_samples.items() 
                                if v.impression.impression_id in sampled_impression_ids}
            self._logger.info(f'Eval on {len(sampled_impression_ids)} impressions ({len(dataset.samples)} samples, ~100% of data)')
        
        dataset.set_mode(Dataset.EVAL_MODE)
        
        # For saving prediction results - use file streaming to avoid OOM
        impression_predictions = {} if save_result else None
        impression_sizes_cache = None
        pred_file = None
        dataloader = None
        
        try:
            if save_result:
                # Pre-compute impression sizes ONCE to avoid O(n) overhead on each flush
                impression_sizes_cache = {}
                for sample in dataset.samples:
                    imp_id = sample.impression.impression_id
                    impression_sizes_cache[imp_id] = impression_sizes_cache.get(imp_id, 0) + 1
                self._logger.info(f'Pre-computed sizes for {len(impression_sizes_cache)} unique impressions')
                
                pred_file = open(os.path.join(self._path, 'prediction_prod.txt'), 'w', encoding='utf-8', buffering=8192*16)
                CleanupHandler.register_file(pred_file)
            
            # Configure DataLoader based on num_workers
            dataloader_kwargs = {
                'batch_size': self.args.eval_batch_size,
                'shuffle': False,
                'num_workers': self.args.dataloader_num_workers,
                'collate_fn': self._collate_fn,
                'drop_last': False,
                'pin_memory': True
            }
            
            # Only add multiprocessing options when num_workers > 0
            if self.args.dataloader_num_workers > 0:
                dataloader_kwargs['persistent_workers'] = True
                dataloader_kwargs['prefetch_factor'] = 2  # Reduced from 4 to save memory
            
            dataloader = DataLoader(dataset, **dataloader_kwargs)
            CleanupHandler.register_dataloader(dataloader)
            total_pos_example = 0
            batch_counter = 0
            
            with torch.no_grad():
                for batch in tqdm(dataloader, total=len(dataloader), desc='Evaluation phase'):
                    # Don't move the entire batch to device yet - we'll handle it in _forward_step_with_cache
                    # Use FP16 for evaluation if enabled
                    if self.args.fp16:
                        with torch.cuda.amp.autocast():
                            poly_attn, logits = self._forward_step_with_cache(model, batch, news_embeddings_cache, encoded_news_ids, debug=False)
                    else:
                        poly_attn, logits = self._forward_step_with_cache(model, batch, news_embeddings_cache, encoded_news_ids, debug=False)
                    
                    # Collect predictions for saving if needed
                    if save_result:
                        probs = torch.sigmoid(logits).cpu()
                        impression_ids = batch['impression_id']
                        candidate_news_ids = batch['candidate_news_ids']
                        
                        for idx in range(len(impression_ids)):
                            imp_id = int(impression_ids[idx][0]) if isinstance(impression_ids[idx], list) else int(impression_ids[idx])
                            score = float(probs[idx][0])
                            news_id = candidate_news_ids[idx][0]
                            
                            if imp_id not in impression_predictions:
                                impression_predictions[imp_id] = []
                            impression_predictions[imp_id].append((news_id, score))
                        
                        # Flush completed impressions to file to save memory
                        # Flush every 1000 batches or when dict size > 10000 to prevent memory buildup
                        if (batch_counter % 30000 == 0 or len(impression_predictions) > 500000) and impression_predictions:
                            flushed_count = self._flush_predictions_to_file(pred_file, impression_predictions, impression_sizes_cache, force_all=False)
                            gc.collect()  # Force garbage collection
                            self._logger.info(f'[BATCH {batch_counter}] Flushed {flushed_count} completed impressions, {len(impression_predictions)} remain in memory')

                    batch_counter += 1

            # Save remaining prediction results (force flush all)
            if save_result:
                if impression_predictions:
                    flushed_count = self._flush_predictions_to_file(pred_file, impression_predictions, impression_sizes_cache, force_all=True)
                    self._logger.info(f'[FINAL] Flushed {flushed_count} remaining impressions')
                
        finally:
            # Cleanup resources to prevent zombie processes
            if pred_file:
                try:
                    pred_file.close()
                    self._logger.info(f'Prediction file saved to: {os.path.join(self._path, "prediction_prod.txt")}')
                except Exception as e:
                    self._logger.error(f'Error closing prediction file: {e}')
            
            # Explicitly cleanup DataLoader workers
            if dataloader is not None and hasattr(dataloader, '_iterator'):
                try:
                    if dataloader._iterator is not None:
                        dataloader._iterator._shutdown_workers()
                except Exception:
                    pass
            
            # Delete dataloader to free workers
            if dataloader is not None:
                del dataloader
            
            # Clear prediction cache
            if impression_predictions:
                impression_predictions.clear()
            del impression_predictions
            
            # Clear CUDA cache to free GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            # Restore original samples if we sampled
            if original_samples is not None:
                dataset._samples = original_samples


    def _train_step(self, batch, model, loss_calculator, accumulation_factor: int):
        model.train()
        batch = utils.to_device(batch, self._device)
        if self.args.fp16:
            with torch.autocast(device_type=self._device.type, dtype=torch.float16):
                poly_attn, logits = self._forward_step(model, batch)
                loss = loss_calculator.compute(poly_attn, logits, batch['label'])
                loss = loss / accumulation_factor
            self.scaler.scale(loss).backward()
        else:
            poly_attn, logits = self._forward_step(model, batch)
            loss = loss_calculator.compute(poly_attn, logits, batch['label'])
            loss = loss / accumulation_factor
            loss.backward()

        return (loss * accumulation_factor).item()

    def _eval(self, model : nn.Module, dataset : Dataset, loss_calculator : Loss, metrics: List[str], save_result: bool = False):
        model.eval()
        
        # Pre-encode ALL unique news FIRST (before sampling) to ensure all news are cached
        self._logger.info('Pre-encoding all unique news for evaluation...')
        news_embeddings_cache, encoded_news_ids = self._encode_all_news(model, dataset)
        self._logger.info(f'Cached {len(encoded_news_ids)} news embeddings')
        
        # Sample 1% of dataset for faster eval during training (AFTER encoding all news)
        original_samples = None
        if not save_result:  # Only sample during training eval, not final eval
            impression_ids = list(set([sample.impression.impression_id for sample in dataset.samples]))
            random.seed(42)  # Reproducible sampling
            sampled_impression_ids = set(random.sample(impression_ids, k=max(1, len(impression_ids) )))
            # Temporarily replace _samples with filtered version
            original_samples = dataset._samples
            dataset._samples = {k: v for k, v in original_samples.items() 
                                if v.impression.impression_id in sampled_impression_ids}
            self._logger.info(f'Eval on {len(sampled_impression_ids)} impressions ({len(dataset.samples)} samples, ~100% of data)')
        
        dataset.set_mode(Dataset.EVAL_MODE)
        if self.args.fast_eval:
            evaluator = FastEvaluator(dataset)
        else:
            evaluator = SlowEvaluator(dataset)
        
        # For saving prediction results - use file streaming to avoid OOM
        impression_predictions = {} if save_result else None
        impression_sizes_cache = None
        pred_file = None
        dataloader = None
        loss = None
        scores = None
        
        try:
            if save_result:
                # Pre-compute impression sizes ONCE to avoid O(n) overhead on each flush
                impression_sizes_cache = {}
                for sample in dataset.samples:
                    imp_id = sample.impression.impression_id
                    impression_sizes_cache[imp_id] = impression_sizes_cache.get(imp_id, 0) + 1
                self._logger.info(f'Pre-computed sizes for {len(impression_sizes_cache)} unique impressions')
                
                pred_file = open(os.path.join(self._path, 'prediction_prod.txt'), 'w', encoding='utf-8', buffering=8192*16)
                CleanupHandler.register_file(pred_file)
            
            # Configure DataLoader based on num_workers
            dataloader_kwargs = {
                'batch_size': self.args.eval_batch_size,
                'shuffle': False,
                'num_workers': self.args.dataloader_num_workers,
                'collate_fn': self._collate_fn,
                'drop_last': False,
                'pin_memory': True
            }
            
            # Only add multiprocessing options when num_workers > 0
            if self.args.dataloader_num_workers > 0:
                dataloader_kwargs['persistent_workers'] = True
                dataloader_kwargs['prefetch_factor'] = 2  # Reduced from 4 to save memory
            
            dataloader = DataLoader(dataset, **dataloader_kwargs)
            CleanupHandler.register_dataloader(dataloader)
            total_loss = 0.0
            total_pos_example = 0
            batch_counter = 0
            prev_batch_end = time.time()
            
            with torch.no_grad():
                for batch in tqdm(dataloader, total=len(dataloader), desc='Evaluation phase'):
                    dataloader_time = time.time() - prev_batch_end
                    batch_start = time.time()
                    # Don't move the entire batch to device yet - we'll handle it in _forward_step_with_cache
                    # Use FP16 for evaluation if enabled
                    if self.args.fp16:
                        with torch.cuda.amp.autocast():
                            poly_attn, logits = self._forward_step_with_cache(model, batch, news_embeddings_cache, encoded_news_ids, debug=False)
                    else:
                        poly_attn, logits = self._forward_step_with_cache(model, batch, news_embeddings_cache, encoded_news_ids, debug=False)
                    forward_end = time.time()
                     
                    if 'loss' in self.args.evaluation_info:
                        batch_loss = loss_calculator.compute_eval_loss(poly_attn, logits, batch['label'].to(self._device))
                        total_loss += batch_loss
                        total_pos_example += batch['label'].sum().item()
                    loss_end = time.time()
                    
                    if 'metrics' in self.args.evaluation_info:
                        evaluator.eval_batch(logits, batch['impression_id'])
                    eval_end = time.time()
                    
                    # Collect predictions for saving if needed
                    if save_result:
                        probs = torch.sigmoid(logits).cpu()
                        impression_ids = batch['impression_id']
                        candidate_news_ids = batch['candidate_news_ids']
                        
                        for idx in range(len(impression_ids)):
                            imp_id = int(impression_ids[idx][0]) if isinstance(impression_ids[idx], list) else int(impression_ids[idx])
                            score = float(probs[idx][0])
                            news_id = candidate_news_ids[idx][0]
                            
                            if imp_id not in impression_predictions:
                                impression_predictions[imp_id] = []
                            impression_predictions[imp_id].append((news_id, score))
                        
                        # Flush completed impressions to file to save memory
                        # Flush every 1000 batches or when dict size > 10000 to prevent memory buildup
                        if (batch_counter % 20000 == 0 or len(impression_predictions) > 500000) and impression_predictions:
                            flushed_count = self._flush_predictions_to_file(pred_file, impression_predictions, impression_sizes_cache, force_all=False)
                            gc.collect()  # Force garbage collection
                            self._logger.info(f'[BATCH {batch_counter}] Flushed {flushed_count} completed impressions, {len(impression_predictions)} remain in memory')
                    
                    """ if batch_counter % 100 == 0:
                        print(f"\n[BATCH TIMING {batch_counter}]")
                        print(f"  ⚠️  DATALOADER WAIT: {dataloader_time*1000:.2f}ms <- BOTTLENECK!")
                        print(f"  Forward pass: {(forward_end-batch_start)*1000:.2f}ms")
                        print(f"  Loss compute: {(loss_end-forward_end)*1000:.2f}ms")
                        print(f"  Evaluator: {(eval_end-loss_end)*1000:.2f}ms")
                        print(f"  TOTAL batch work: {(eval_end-batch_start)*1000:.2f}ms") """
                    batch_counter += 1
                    prev_batch_end = time.time()

            if 'loss' in self.args.evaluation_info:
                loss = total_loss / total_pos_example
            else:
                loss = None
            if 'metrics' in self.args.evaluation_info:
                scores = evaluator.compute_scores(metrics, save_result, self._path)
            else:
                scores = None

            # Save remaining prediction results (force flush all)
            if save_result:
                if impression_predictions:
                    flushed_count = self._flush_predictions_to_file(pred_file, impression_predictions, impression_sizes_cache, force_all=True)
                    self._logger.info(f'[FINAL] Flushed {flushed_count} remaining impressions')
                
        finally:
            # Cleanup resources to prevent zombie processes
            if pred_file:
                try:
                    pred_file.close()
                    self._logger.info(f'Prediction file saved to: {os.path.join(self._path, "prediction_prod.txt")}')
                except Exception as e:
                    self._logger.error(f'Error closing prediction file: {e}')
            
            # Explicitly cleanup DataLoader workers
            if dataloader is not None and hasattr(dataloader, '_iterator'):
                try:
                    if dataloader._iterator is not None:
                        dataloader._iterator._shutdown_workers()
                except Exception:
                    pass
            
            # Delete dataloader to free workers
            if dataloader is not None:
                del dataloader
            
            # Clear prediction cache
            if impression_predictions:
                impression_predictions.clear()
            del impression_predictions
            
            # Clear evaluator cache
            if 'evaluator' in locals():
                del evaluator
            
            # Clear CUDA cache to free GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            # Restore original samples if we sampled
            if original_samples is not None:
                dataset._samples = original_samples

        return loss, scores

    @staticmethod
    def _create_loss():
        criterion = nn.CrossEntropyLoss(reduction='mean')
        loss_calculator = Loss(criterion)

        return loss_calculator

    def _flush_predictions_to_file(self, file_handle, impression_predictions, impression_sizes_cache, force_all=False):
        """
        Write impressions to file and clear from memory.
        
        Args:
            file_handle: open file handle to write to
            impression_predictions: dict of impression_id -> [(news_id, score), ...]
            impression_sizes_cache: pre-computed dict of impression_id -> expected size
            force_all: if True, write all impressions regardless of completion status
        """
        if not impression_predictions:
            return 0
        
        # Write impressions
        flushed_imps = []
        for imp_id, predictions in impression_predictions.items():
            # Check if impression is complete
            expected_size = impression_sizes_cache.get(imp_id, len(predictions))
            is_complete = len(predictions) >= expected_size
            
            # Write if complete OR if force_all is True
            if is_complete or force_all:
                # Write scores for this impression
                scores = [score for _, score in predictions]
                scores_str = ','.join([f'{s:.6f}' for s in scores])
                file_handle.write(f'{imp_id} [{scores_str}]\n')
                flushed_imps.append(imp_id)
        
        # Remove flushed impressions from memory
        for imp_id in flushed_imps:
            del impression_predictions[imp_id]
        
        # Flush to disk immediately
        if file_handle:
            file_handle.flush()
            os.fsync(file_handle.fileno())
        
        return len(flushed_imps)

    def _save_prediction_file(self, impression_predictions):
        """
        Save prediction results in CodaLab submission format.
        
        Format: ImpressionID [Rank-of-News1,Rank-of-News2,...,Rank-of-NewsN]
        Where ranks indicate the ranking order (1=best, higher=worse)
        
        Also saves prediction_score.txt with actual scores before ranking.
        
        Args:
            impression_predictions: dict mapping impression_id to list of (news_id, score) tuples
        """
        # Save ranking file (prediction.txt)
        output_path = os.path.join(self._path, 'prediction.txt')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for imp_id in sorted(impression_predictions.keys()):
                news_scores = impression_predictions[imp_id]
                
                # Sort by score (descending) to get ranking
                # Higher scores get better (lower) ranks
                sorted_indices = sorted(range(len(news_scores)), 
                                      key=lambda i: news_scores[i][1], 
                                      reverse=True)
                
                # Create rank list: position in original order gets its rank
                ranks = [0] * len(sorted_indices)
                for rank, idx in enumerate(sorted_indices, start=1):
                    ranks[idx] = rank
                
                # Format: impression_id [rank1,rank2,rank3,...]
                ranks_str = ','.join(map(str, ranks))
                f.write(f'{imp_id} [{ranks_str}]\n')
        
        self._logger.info(f'Prediction file saved to: {output_path}')
        
        # Save score file (prediction_score.txt)
        score_output_path = os.path.join(self._path, 'prediction_score.txt')
        
        with open(score_output_path, 'w', encoding='utf-8') as f:
            for imp_id in sorted(impression_predictions.keys()):
                news_scores = impression_predictions[imp_id]
                
                # Format: impression_id [score1,score2,score3,...]
                scores = [score for _, score in news_scores]
                scores_str = ','.join([f'{s:.6f}' for s in scores])
                f.write(f'{imp_id} [{scores_str}]\n')
        
        self._logger.info(f'Prediction score file saved to: {score_output_path}')
        self._logger.info(f'Total impressions: {len(impression_predictions)}')

    def _collate_fn(self, batch):
        padded_batch = dict()
        keys = batch[0].keys()
        for key in keys:
            samples = [s[key] for s in batch]
            if key in ['his_news_ids', 'candidate_news_ids']:
                # Keep as list of lists for news IDs
                padded_batch[key] = samples
            elif isinstance(samples[0], torch.Tensor):
                if not samples[0].shape:
                    # Scalar tensor
                    padded_batch[key] = torch.stack(samples)
                else:
                    # Multi-dimensional tensor - needs padding
                    if key in ['his_title', 'title', 'his_sapo', 'sapo']:
                        padded_batch[key] = utils.padded_stack(samples, padding=self._tokenizer.pad_token_id)
                    elif key in ['his_category', 'category']:
                        padded_batch[key] = utils.padded_stack(samples, padding=self._category2id['pad'])
                    else:
                        padded_batch[key] = utils.padded_stack(samples, padding=0)
            else:
                # Non-tensor data (should not happen, but handle gracefully)
                padded_batch[key] = samples

        return padded_batch

    def _get_optimizer_params(self, model):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_params = [{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                             'weight_decay': self.args.weight_decay},
                            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                             'weight_decay': 0.0}]

        return optimizer_params

    def _encode_all_news(self, model, dataset):
        """Pre-encode all unique news in the dataset and cache embeddings as tensor"""
        unique_news = dataset.get_all_unique_news()
        max_news_id = max(news.news_id for news in unique_news)
        
        # Handle DataParallel wrapper - extract the underlying model
        actual_model = model.module if isinstance(model, nn.DataParallel) else model
        
        # Create tensor-based cache for O(1) indexing instead of dict lookup
        embedding_dim = 256 if hasattr(self.args, 'word_embed_dim') else 768
        news_embeddings_tensor = torch.zeros(max_news_id + 1, embedding_dim, dtype=torch.float32)
        # Track which news IDs have been encoded (tensor doesn't support 'in' operator)
        encoded_news_ids = set()
        
        # Batch encode for efficiency
        batch_size = self.args.eval_batch_size
        num_batches = (len(unique_news) + batch_size - 1) // batch_size
        
        try:
            for i in tqdm(range(0, len(unique_news), batch_size), 
                          total=num_batches, 
                          desc='Encoding news embeddings'):
                batch_news = unique_news[i:i + batch_size]
                
                # Prepare batch
                titles = [news.title for news in batch_news]
                sapos = [news.sapo for news in batch_news]
                
                titles = utils.padded_stack(titles, padding=self._tokenizer.pad_token_id).to(self._device)
                sapos = utils.padded_stack(sapos, padding=self._tokenizer.pad_token_id).to(self._device)
                title_masks = (titles != self._tokenizer.pad_token_id)
                sapo_masks = (sapos != self._tokenizer.pad_token_id)
                
                # Encode using the actual model (not the wrapper)
                if self.args.fp16:
                    try:
                        with torch.amp.autocast('cuda'):
                            embeddings = actual_model.news_encoder(title_encoding=titles, title_attn_mask=title_masks,
                                                           sapo_encoding=sapos, sapo_attn_mask=sapo_masks)
                    except AttributeError:
                        with torch.cuda.amp.autocast():
                            embeddings = actual_model.news_encoder(title_encoding=titles, title_attn_mask=title_masks,
                                                           sapo_encoding=sapos, sapo_attn_mask=sapo_masks)
                else:
                    embeddings = actual_model.news_encoder(title_encoding=titles, title_attn_mask=title_masks,
                                                  sapo_encoding=sapos, sapo_attn_mask=sapo_masks)
                
                # Cache embeddings in tensor - stay on CPU to save GPU memory
                for j, news in enumerate(batch_news):
                    news_embeddings_tensor[news.news_id] = embeddings[j].detach().cpu()
                    encoded_news_ids.add(news.news_id)  # Track encoded IDs
                
                # Clear GPU cache periodically
                if (i // batch_size) % 50 == 0:
                    torch.cuda.empty_cache()
        
        finally:
            # Final cleanup
            torch.cuda.empty_cache()
            gc.collect()
        
        return (news_embeddings_tensor, encoded_news_ids)
    
    def _forward_step_with_cache(self, model, batch, news_embeddings_cache, encoded_news_ids, debug=False):
        """Forward step using pre-cached news embeddings - OPTIMIZED with tensor indexing"""
        from src.utils import pairwise_cosine_similarity
        import time
        
        # Handle DataParallel wrapper - extract the underlying model
        actual_model = model.module if isinstance(model, nn.DataParallel) else model
        
        batch_size = batch['his_title'].shape[0]
        num_candidates = batch['title'].shape[1]
        his_length = batch['his_title'].shape[1]
        
        if debug:
            self._logger.info(f"\n[DEBUG _forward_step_with_cache]")
            self._logger.info(f"  batch_size: {batch_size}")
            self._logger.info(f"  num_candidates: {num_candidates}")
            self._logger.info(f"  his_length: {his_length}")
        
        # Move category tensors to device (needed for category bias)
        t0 = time.time()
        his_category = batch['his_category'].to(self._device)
        
        # CRITICAL FIX: Calculate actual num_candidates from news_ids first
        # to properly slice category tensor (avoid using padded values)
        total_cand_ids = sum(len(ids) for ids in batch['candidate_news_ids'])
        actual_num_candidates_pre = total_cand_ids // batch_size
        
        # Slice category to match actual candidates (not padded)
        candidate_category = batch['category'][:, :actual_num_candidates_pre].to(self._device)
        his_mask = batch['his_mask'].to(self._device)
        t1 = time.time()
        
        # OPTIMIZED: Vectorized batch lookup for history news
        # Collect all news IDs first, then batch transfer
        all_his_ids = []
        for sample_his_ids in batch['his_news_ids']:
            all_his_ids.extend(sample_his_ids)
        
        t2 = time.time()
        # Vectorized lookup using advanced indexing (MUCH FASTER)
        # news_embeddings_cache is already a tensor, so we can use fancy indexing
        all_his_embeddings = news_embeddings_cache[all_his_ids]  # Single CPU tensor operation
        t3 = time.time()
        
        # Single batched transfer CPU→GPU (CRITICAL OPTIMIZATION)
        all_his_embeddings_gpu = all_his_embeddings.to(self._device, non_blocking=True)
        
        # Reshape to (batch_size, his_length, embed_dim)
        history_repr = all_his_embeddings_gpu.view(batch_size, his_length, -1)
        t4 = time.time()
        
        """ if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 100 == 0:
            print(f"\n[DEBUG Batch {self._debug_counter}]")
            print(f"  Category to device: {(t1-t0)*1000:.2f}ms")
            print(f"  Collect his_ids: {(t2-t1)*1000:.2f}ms")
            print(f"  Index cache (HIS): {(t3-t2)*1000:.2f}ms <- CHECK THIS")
            print(f"  Transfer to GPU: {(t4-t3)*1000:.2f}ms") """
        
        # OPTIMIZED: Vectorized batch lookup for candidate news
        all_cand_ids = []
        for sample_cand_ids in batch['candidate_news_ids']:
            all_cand_ids.extend(sample_cand_ids)
        
        # CRITICAL FIX: Calculate actual num_candidates from news_ids, not padded tensor
        # In submission mode, each sample has only 1 candidate, so len(all_cand_ids) / batch_size = 1
        actual_num_candidates = len(all_cand_ids) // batch_size
        
        if debug:
            self._logger.info(f"  First 10 cand_ids: {all_cand_ids[:10]}")
            self._logger.info(f"  Unique cand_ids in batch: {len(set(all_cand_ids))}")
            self._logger.info(f"  actual_num_candidates (from IDs): {actual_num_candidates}")
            self._logger.info(f"  num_candidates (from padded tensor): {num_candidates}")
        
        t5 = time.time()
        # Vectorized lookup using advanced indexing
        all_cand_embeddings = news_embeddings_cache[all_cand_ids]  # Single CPU tensor operation
        t6 = time.time()
        
        # Single batched transfer CPU→GPU (CRITICAL OPTIMIZATION)
        all_cand_embeddings_gpu = all_cand_embeddings.to(self._device, non_blocking=True)
        
        # Reshape to (batch_size, actual_num_candidates, embed_dim) - use actual count!
        candidate_repr = all_cand_embeddings_gpu.view(batch_size, actual_num_candidates, -1)
        
        if debug:
            self._logger.info(f"  candidate_repr shape: {candidate_repr.shape}")
            self._logger.info(f"  First 5 candidate embeddings norm: {[candidate_repr[i, 0].norm().item() for i in range(min(5, batch_size))]}")
        
        t7 = time.time()
        
        """ if self._debug_counter % 100 == 0:
            print(f"  Index cache (CAND): {(t6-t5)*1000:.2f}ms <- CHECK THIS")
            print(f"  Transfer CAND to GPU: {(t7-t6)*1000:.2f}ms")
            print(f"  Cache size: {news_embeddings_cache.shape}, dtype: {news_embeddings_cache.dtype}")
            print(f"  His IDs range: [{min(all_his_ids)}, {max(all_his_ids)}], count: {len(all_his_ids)}")
            print(f"  Cand IDs range: [{min(all_cand_ids)}, {max(all_cand_ids)}], count: {len(all_cand_ids)}") """
        
        t8 = time.time()
        # Category bias and poly attention - use actual_model
        if actual_model.use_category_bias:
            his_category_embed = actual_model.category_embedding(his_category)
            his_category_embed = actual_model.category_dropout(his_category_embed)
            candidate_category_embed = actual_model.category_embedding(candidate_category)
            candidate_category_embed = actual_model.category_dropout(candidate_category_embed)
            category_bias = pairwise_cosine_similarity(his_category_embed, candidate_category_embed)
            multi_user_interest = actual_model.poly_attn(embeddings=history_repr, attn_mask=his_mask, bias=category_bias)
        else:
            multi_user_interest = actual_model.poly_attn(embeddings=history_repr, attn_mask=his_mask, bias=None)
        t9 = time.time()
        
        # Click predictor - use actual_model
        matching_scores = torch.matmul(candidate_repr, multi_user_interest.permute(0, 2, 1))
        if actual_model.score_type == 'max':
            matching_scores = matching_scores.max(dim=2)[0]
        elif actual_model.score_type == 'mean':
            matching_scores = matching_scores.mean(dim=2)
        elif actual_model.score_type == 'weighted':
            matching_scores = actual_model.target_aware_attn(query=multi_user_interest, key=candidate_repr, value=matching_scores)
        
        if debug:
            self._logger.info(f"  matching_scores shape: {matching_scores.shape}")
            self._logger.info(f"  First 10 matching_scores: {matching_scores[:10, 0].cpu().numpy()}")
        
        t10 = time.time()
        
        """ if self._debug_counter % 100 == 0:
            print(f"  Poly attention: {(t9-t8)*1000:.2f}ms")
            print(f"  Matching scores: {(t10-t9)*1000:.2f}ms")
            print(f"  TOTAL forward: {(t10-t0)*1000:.2f}ms") """
        
        return multi_user_interest, matching_scores
    
    @staticmethod
    def _forward_step(model, batch):
        poly_attn, logits = model(title=batch['title'], title_mask=batch['title_mask'], his_title=batch['his_title'],
                                  his_title_mask=batch['his_title_mask'], his_mask=batch['his_mask'],
                                  sapo=batch['sapo'], sapo_mask=batch['sapo_mask'], his_sapo=batch['his_sapo'],
                                  his_sapo_mask=batch['his_sapo_mask'], category=batch['category'],
                                  his_category=batch['his_category'])
        return poly_attn, logits

    
    def _build_model(self, category_embed):
        """Build model from args - needed for loading new format checkpoints"""
        args = self.args
        
        config = RobertaConfig.from_pretrained(args.pretrained_embedding)
        news_encoder = NewsEncoder.from_pretrained(args.pretrained_embedding, config=config,
                                                   apply_reduce_dim=args.apply_reduce_dim, use_sapo=args.use_sapo,
                                                   dropout=args.dropout, freeze_transformer=args.freeze_transformer,
                                                   word_embed_dim=args.word_embed_dim, combine_type=args.combine_type,
                                                   lstm_num_layers=args.lstm_num_layers, lstm_dropout=args.lstm_dropout)
        model = Miner(news_encoder=news_encoder, use_category_bias=args.use_category_bias,
                      num_context_codes=args.num_context_codes, context_code_dim=args.context_code_dim,
                      score_type=args.score_type, dropout=args.dropout, num_category=len(self._category2id),
                      category_embed_dim=args.category_embed_dim, category_pad_token_id=self._category2id['pad'],
                      category_embed=category_embed)
        
        return model