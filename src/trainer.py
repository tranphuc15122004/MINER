import json
import math
import os
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
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                self._logger.info(f'Model was trained for {checkpoint["epoch"]} epochs')
        model.to(self._device)
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
                                  save_result=args.save_eval_result)
        if 'loss' in self.args.evaluation_info:
            self._logger.info('Loss {}'.format(loss))
        for metric in args.metrics:
            self._logger.info(f'Metric {metric}: {scores[metric]}')
    
    def submission_generator(self):
        args = self.args
        self._log_arguments()

        # Load model
        checkpoint = self._load_model(args.saved_model_path)
        
        if 'is_old_format' in checkpoint:
            model = checkpoint['model']
            if 'epoch' in checkpoint:
                self._logger.info(f'Model was trained for {checkpoint.get("epoch", "unknown")} epochs')
        else:
            self._logger.info('Rebuilding model from checkpoint...')
            category_embed = None
            model = self._build_model(category_embed)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                self._logger.info(f'Model was trained for {checkpoint["epoch"]} epochs')
        
        model.to(self._device)
        model.eval()

        # Read submission dataset (no labels)
        reader = Reader(tokenizer=self._tokenizer, max_title_length=args.max_title_length,
                        max_sapo_length=args.max_sapo_length, user2id=self._user2id, category2id=self._category2id,
                        max_his_click=args.his_length, npratio=None)
        dataset = reader.read_submission_dataset(args.data_name, args.eval_news_path, args.eval_behaviors_path)
        self._logger.info(f'Model: {self.args.model_name}')
        self._logger.info(f'Dataset: {self.args.data_name}')
        self._logger.info(f'Test dataset: {len(dataset)} samples')

        # Submission generation
        self._logger.info('----------------  Generation phrase  ----------------')
        
        # Pre-encode all news for faster inference
        self._logger.info('Pre-encoding all unique news...')
        news_embeddings_cache = self._encode_all_news(model, dataset)
        self._logger.info(f'Cached {len(news_embeddings_cache)} news embeddings')
        
        # Check if we should use full dataset or sample (default: sample 1%)
        use_full_dataset = getattr(args, 'use_full_dataset', False)
        
        if not use_full_dataset:
            # Test mode: Sample 1% impressions for quick testing
            impression_ids = list(set([sample.impression.impression_id for sample in dataset.samples]))
            random.seed(42)
            sampled_impression_ids = set(random.sample(impression_ids, k=max(1, len(impression_ids) // 100)))
            original_samples = dataset._samples
            dataset._samples = {k: v for k, v in original_samples.items() 
                                if v.impression.impression_id in sampled_impression_ids}
            self._logger.info(f'Sampling mode: Using {len(sampled_impression_ids)} impressions ({len(dataset.samples)} samples, ~1% of data)')
        else:
            # Full dataset mode (for production submission)
            self._logger.info(f'Full dataset mode: Using all {len(dataset.samples)} samples')
            original_samples = None
        
        dataset.set_mode(Dataset.EVAL_MODE)
        dataloader = DataLoader(dataset, batch_size=self.args.eval_batch_size, shuffle=False,
                                num_workers=self.args.dataloader_num_workers, collate_fn=self._collate_fn,
                                drop_last=False)
        
        # Collect predictions grouped by impression_id
        impression_predictions = {}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader), desc='Generating predictions'):
                if self.args.fp16:
                    with torch.cuda.amp.autocast():
                        _, logits = self._forward_step_with_cache(model, batch, news_embeddings_cache)
                else:
                    _, logits = self._forward_step_with_cache(model, batch, news_embeddings_cache)
                
                # Get scores (use sigmoid for independent probability estimation)
                scores = torch.sigmoid(logits).cpu().numpy()
                impression_ids = batch['impression_id'].cpu().numpy()
                candidate_news_ids = batch['candidate_news_ids']
                
                # Group predictions by impression_id
                for idx in range(len(impression_ids)):
                    imp_id = int(impression_ids[idx])
                    score = float(scores[idx][0])  # Single candidate per sample
                    news_id = candidate_news_ids[idx][0]  # Single news per sample
                    
                    if imp_id not in impression_predictions:
                        impression_predictions[imp_id] = []
                    impression_predictions[imp_id].append((news_id, score))
        
        # Restore original samples if we sampled
        if original_samples is not None:
            dataset._samples = original_samples
        
        # Sort and generate submission file in MIND format
        self._logger.info('Generating submission file...')
        output_path = os.path.join(self._path, 'prediction_rank.txt')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for imp_id in sorted(impression_predictions.keys()):
                news_scores = impression_predictions[imp_id]
                
                # Rank mode: Output ranks (for CodaLab submission)
                sorted_indices = sorted(range(len(news_scores)), 
                                        key=lambda i: news_scores[i][1], 
                                        reverse=True)
                ranks = [0] * len(sorted_indices)
                for rank, idx in enumerate(sorted_indices, start=1):
                    ranks[idx] = rank
                
                # Format: impression_id [rank1,rank2,rank3,...]
                output_str_rank = ','.join(map(str, ranks))
            
                f.write(f'{imp_id} [{output_str_rank}]\n')
        
        self._logger.info(f'Submission file saved to: {output_path}')
        self._logger.info(f'Total impressions: {len(impression_predictions)}')
        
        
        output_path = os.path.join(self._path, 'prediction_prod.txt')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for imp_id in sorted(impression_predictions.keys()):
                news_scores = impression_predictions[imp_id]
                
                # Prob mode: Output probabilities
                # Format: impression_id [prob1,prob2,prob3,...]
                probs = [score for _, score in news_scores]
                output_str_prod = ','.join([f'{p:.6f}' for p in probs])
            
                f.write(f'{imp_id} [{output_str_prod}]\n')
        
        self._logger.info(f'Submission file saved to: {output_path}')
        self._logger.info(f'Total impressions: {len(impression_predictions)}')
                

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
        
        # Sample 1% of dataset for faster eval during training
        original_samples = None
        if not save_result:  # Only sample during training eval, not final eval
            impression_ids = list(set([sample.impression.impression_id for sample in dataset.samples]))
            random.seed(42)  # Reproducible sampling
            sampled_impression_ids = set(random.sample(impression_ids, k=max(1, len(impression_ids) // 2)))
            # Temporarily replace _samples with filtered version
            original_samples = dataset._samples
            dataset._samples = {k: v for k, v in original_samples.items() 
                                if v.impression.impression_id in sampled_impression_ids}
            self._logger.info(f'Eval on {len(sampled_impression_ids)} impressions ({len(dataset.samples)} samples, ~50% of data)')
        
        # Pre-encode all news for faster evaluation
        self._logger.info('Pre-encoding all unique news for evaluation...')
        news_embeddings_cache = self._encode_all_news(model, dataset)
        self._logger.info(f'Cached {len(news_embeddings_cache)} news embeddings')
        
        dataset.set_mode(Dataset.EVAL_MODE)
        if self.args.fast_eval:
            evaluator = FastEvaluator(dataset)
        else:
            evaluator = SlowEvaluator(dataset)
        
        dataloader = DataLoader(dataset, batch_size=self.args.eval_batch_size, shuffle=False,
                                num_workers=self.args.dataloader_num_workers, collate_fn=self._collate_fn,
                                drop_last=False)
        total_loss = 0.0
        total_pos_example = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader), desc='Evaluation phase'):
                # Don't move the entire batch to device yet - we'll handle it in _forward_step_with_cache
                # Use FP16 for evaluation if enabled
                if self.args.fp16:
                    with torch.cuda.amp.autocast():
                        poly_attn, logits = self._forward_step_with_cache(model, batch, news_embeddings_cache)
                else:
                    poly_attn, logits = self._forward_step_with_cache(model, batch, news_embeddings_cache)
                if 'loss' in self.args.evaluation_info:
                    batch_loss = loss_calculator.compute_eval_loss(poly_attn, logits, batch['label'].to(self._device))
                    total_loss += batch_loss
                    total_pos_example += batch['label'].sum().item()
                if 'metrics' in self.args.evaluation_info:
                    evaluator.eval_batch(logits, batch['impression_id'])

        if 'loss' in self.args.evaluation_info:
            loss = total_loss / total_pos_example
        else:
            loss = None
        if 'metrics' in self.args.evaluation_info:
            scores = evaluator.compute_scores(metrics, save_result, self._path)
        else:
            scores = None

        # Restore original samples if we sampled
        if original_samples is not None:
            dataset._samples = original_samples

        return loss, scores

    @staticmethod
    def _create_loss():
        criterion = nn.CrossEntropyLoss(reduction='mean')
        loss_calculator = Loss(criterion)

        return loss_calculator

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
        
        # Create tensor-based cache for O(1) indexing instead of dict lookup
        embedding_dim = 256 if hasattr(self.args, 'word_embed_dim') else 768
        news_embeddings_tensor = torch.zeros(max_news_id + 1, embedding_dim, dtype=torch.float32)
        
        # Batch encode for efficiency
        batch_size = self.args.eval_batch_size
        num_batches = (len(unique_news) + batch_size - 1) // batch_size
        
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
            
            # Encode
            if self.args.fp16:
                try:
                    with torch.amp.autocast('cuda'):
                        embeddings = model.news_encoder(title_encoding=titles, title_attn_mask=title_masks,
                                                       sapo_encoding=sapos, sapo_attn_mask=sapo_masks)
                except AttributeError:
                    with torch.cuda.amp.autocast():
                        embeddings = model.news_encoder(title_encoding=titles, title_attn_mask=title_masks,
                                                       sapo_encoding=sapos, sapo_attn_mask=sapo_masks)
            else:
                embeddings = model.news_encoder(title_encoding=titles, title_attn_mask=title_masks,
                                              sapo_encoding=sapos, sapo_attn_mask=sapo_masks)
            
            # Cache embeddings in tensor - stay on CPU to save GPU memory
            for j, news in enumerate(batch_news):
                news_embeddings_tensor[news.news_id] = embeddings[j].detach().cpu()
            
            # Clear GPU cache periodically
            if (i // batch_size) % 50 == 0:
                torch.cuda.empty_cache()
        
        return news_embeddings_tensor
    
    def _forward_step_with_cache(self, model, batch, news_embeddings_cache):
        """Forward step using pre-cached news embeddings - OPTIMIZED with tensor indexing"""
        from src.utils import pairwise_cosine_similarity
        
        batch_size = batch['his_title'].shape[0]
        num_candidates = batch['title'].shape[1]
        his_length = batch['his_title'].shape[1]
        
        # Move category tensors to device (needed for category bias)
        his_category = batch['his_category'].to(self._device)
        candidate_category = batch['category'].to(self._device)
        his_mask = batch['his_mask'].to(self._device)
        
        # Batch lookup cached embeddings for history news
        # Shape: (batch_size, his_length, embed_dim)
        history_repr_list = []
        for idx, sample_his_ids in enumerate(batch['his_news_ids']):
            # Collect all embeddings for this sample first (stay on CPU)
            sample_history_cpu = []
            missing_indices = []
            
            for i, news_id in enumerate(sample_his_ids):
                if news_id in news_embeddings_cache:
                    sample_history_cpu.append(news_embeddings_cache[news_id])
                else:
                    missing_indices.append(i)
                    sample_history_cpu.append(None)  # Placeholder
            
            # Encode missing news if any
            if missing_indices:
                print('         Encode a missing NEWS           ')
                for i in missing_indices:
                    title = batch['his_title'][idx][i].unsqueeze(0).to(self._device)
                    sapo = batch['his_sapo'][idx][i].unsqueeze(0).to(self._device)
                    title_mask = batch['his_title_mask'][idx][i].unsqueeze(0).to(self._device)
                    sapo_mask = batch['his_sapo_mask'][idx][i].unsqueeze(0).to(self._device)
                    
                    if self.args.fp16:
                        with torch.cuda.amp.autocast():
                            embedding = model.news_encoder(title_encoding=title, title_attn_mask=title_mask,
                                                          sapo_encoding=sapo, sapo_attn_mask=sapo_mask)
                    else:
                        embedding = model.news_encoder(title_encoding=title, title_attn_mask=title_mask,
                                                      sapo_encoding=sapo, sapo_attn_mask=sapo_mask)
                    news_id = sample_his_ids[i]
                    sample_history_cpu[i] = embedding.squeeze(0).detach().cpu()
                    news_embeddings_cache[news_id] = sample_history_cpu[i]
            
            # Batch transfer CPU→GPU: stack then move (much faster!)
            sample_history_gpu = torch.stack(sample_history_cpu).to(self._device)
            history_repr_list.append(sample_history_gpu)
        
        history_repr = torch.stack(history_repr_list)  # (batch_size, his_length, embed_dim)
        
        # Batch lookup cached embeddings for candidate news
        # Shape: (batch_size, num_candidates, embed_dim)
        candidate_repr_list = []
        for idx, sample_cand_ids in enumerate(batch['candidate_news_ids']):
            # Collect all embeddings for this sample first (stay on CPU)
            sample_candidates_cpu = []
            missing_indices = []
            
            for i, news_id in enumerate(sample_cand_ids):
                if news_id in news_embeddings_cache:
                    sample_candidates_cpu.append(news_embeddings_cache[news_id])
                else:
                    missing_indices.append(i)
                    sample_candidates_cpu.append(None)  # Placeholder
            
            # Encode missing news if any
            if missing_indices:
                for i in missing_indices:
                    title = batch['title'][idx][i].unsqueeze(0).to(self._device)
                    sapo = batch['sapo'][idx][i].unsqueeze(0).to(self._device)
                    title_mask = batch['title_mask'][idx][i].unsqueeze(0).to(self._device)
                    sapo_mask = batch['sapo_mask'][idx][i].unsqueeze(0).to(self._device)
                    
                    if self.args.fp16:
                        with torch.cuda.amp.autocast():
                            embedding = model.news_encoder(title_encoding=title, title_attn_mask=title_mask,
                                                          sapo_encoding=sapo, sapo_attn_mask=sapo_mask)
                    else:
                        embedding = model.news_encoder(title_encoding=title, title_attn_mask=title_mask,
                                                      sapo_encoding=sapo, sapo_attn_mask=sapo_mask)
                    news_id = sample_cand_ids[i]
                    sample_candidates_cpu[i] = embedding.squeeze(0).detach().cpu()
                    news_embeddings_cache[news_id] = sample_candidates_cpu[i]
            
            # Batch transfer CPU→GPU: stack then move (much faster!)
            sample_candidates_gpu = torch.stack(sample_candidates_cpu).to(self._device)
            candidate_repr_list.append(sample_candidates_gpu)
        
        candidate_repr = torch.stack(candidate_repr_list)  # (batch_size, num_candidates, embed_dim)
        
        # Category bias and poly attention (SAME AS ORIGINAL)
        if model.use_category_bias:
            his_category_embed = model.category_embedding(his_category)
            his_category_embed = model.category_dropout(his_category_embed)
            candidate_category_embed = model.category_embedding(candidate_category)
            candidate_category_embed = model.category_dropout(candidate_category_embed)
            category_bias = pairwise_cosine_similarity(his_category_embed, candidate_category_embed)
            multi_user_interest = model.poly_attn(embeddings=history_repr, attn_mask=his_mask, bias=category_bias)
        else:
            multi_user_interest = model.poly_attn(embeddings=history_repr, attn_mask=his_mask, bias=None)
        
        # Click predictor (SAME AS ORIGINAL)
        matching_scores = torch.matmul(candidate_repr, multi_user_interest.permute(0, 2, 1))
        if model.score_type == 'max':
            matching_scores = matching_scores.max(dim=2)[0]
        elif model.score_type == 'mean':
            matching_scores = matching_scores.mean(dim=2)
        elif model.score_type == 'weighted':
            matching_scores = model.target_aware_attn(query=multi_user_interest, key=candidate_repr, value=matching_scores)
        
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