# MINER - Multi-Interest News Recommendation

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Triển khai mô hình **MINER (Multi-Interest Network for News Recommendation)** dựa trên paper [ACL 2022 Findings](https://aclanthology.org/2022.findings-acl.29.pdf), áp dụng cho bài toán gợi ý tin tức trên **MIND Large dataset**.

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Kiến trúc mô hình](#-kiến-trúc-mô-hình)
- [Cài đặt](#-cài-đặt)
- [Chuẩn bị dữ liệu](#-chuẩn-bị-dữ-liệu)
- [Sử dụng](#-sử-dụng)
  - [Training](#1-training)
  - [Evaluation](#2-evaluation)
  - [Submission Generation](#3-submission-generation)
- [Ensemble Methods](#-ensemble-methods)
- [Kết quả](#-kết-quả)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Tài liệu tham khảo](#-tài-liệu-tham-khảo)

## 🎯 Tổng quan

MINER là mô hình neural network cho bài toán **news recommendation**, với các đặc điểm:
- **Multi-interest user modeling**: Học K interests khác nhau từ lịch sử đọc bằng Poly-Attention
- **Category-aware attention**: Dùng category embedding làm bias để cải thiện news representation
- **Flexible score aggregation**: Hỗ trợ `mean`, `max`, `weighted` để tổng hợp K matching scores
- **Ensemble learning**: Kết hợp predictions từ nhiều models (Weighted Mean & Stacking)
- **Production-ready inference**: Hỗ trợ inference không cần ground truth

### Tính năng chính

- **News Encoder**: DistilRoBERTa encode title (và sapo nếu bật `--use_sapo`), reduce xuống 256 chiều qua Linear
- **Category Embedding**: Embed category (300 chiều) làm attention bias (bật bằng `--use_category_bias`)
- **Sapo Combine**: Kết hợp title + sapo qua `linear` hoặc `lstm`
- **Poly-Attention User Encoder**: K=32 interest vectors, mỗi vector dot-product với candidate news
- **Gradient Accumulation + fp16**: Hỗ trợ training trên GPU bộ nhớ thấp


## 🔧 Cài đặt

### Yêu cầu hệ thống

- Python >= 3.8
- CUDA (khuyến nghị cho training)
- RAM >= 16GB
- Disk >= 50GB (cho MIND Large dataset)

### Cài đặt dependencies

```bash
# Clone repository
git clone https://github.com/tranphuc15122004/MINER.git
cd MINER

# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt
```

### Dependencies chính

| Package | Version | Mục đích |
|---------|---------|---------|
| `torch` | 2.1.0 | Deep learning framework |
| `transformers` | 4.37.2 | DistilRoBERTa & tokenizer |
| `accelerate` | 0.27.2 | Mixed precision training |
| `scikit-learn` | 1.3.2 | Stacking ensemble meta-model |
| `pandas` | 2.1.3 | Data processing |
| `tensorboard` | 2.15.1 | Training visualization |
| `tqdm` | 4.66.1 | Progress bars |

## 📊 Chuẩn bị dữ liệu

### 1. Download MIND Large dataset

```bash
# Train set
wget https://mind201910.blob.core.windows.net/release/MINDlarge_train.zip
unzip MINDlarge_train.zip -d data/MINDlarge_train/

# Dev set (dùng làm validation)
wget https://mind201910.blob.core.windows.net/release/MINDlarge_dev.zip
unzip MINDlarge_dev.zip -d data/MINDlarge_dev/

# Test set (không có labels, dùng để tạo submission)
wget https://mind201910.blob.core.windows.net/release/MINDlarge_test.zip
unzip MINDlarge_test.zip -d data/MINDlarge_test/
```

Sau khi giải nén, copy `behaviors.tsv` và `news.tsv` từ dev set vào `data/valid/`:
```bash
cp data/MINDlarge_dev/MINDlarge_dev/behaviors.tsv data/valid/
cp data/MINDlarge_dev/MINDlarge_dev/news.tsv data/valid/
```

### 2. Tạo mappings

```bash
python prepare_mind_mappings.py
```

Script tự động đọc từ `data/MINDlarge_train/` và `data/MINDlarge_dev/`, tạo ra:
- `data/user2id.json` - Mapping user ID → integer (bao gồm `pad` token)
- `data/category2id.json` - Mapping category → integer

## 🚀 Sử dụng

### 1. Training

#### Option A: Dùng config file (khuyến nghị)

```bash
python main.py train @config/train.txt
```

Config mặc định (`config/train.txt`) sử dụng:
- PLM: `distilroberta-base`
- title: 20 tokens, sapo: 64 tokens, lịch sử: 50 bài
- K=32 interest codes, context_code_dim=200, word_embed_dim=256
- batch_size=12, gradient_accumulation=32 (effective batch=384)
- fp16, cosine lr scheduler, 3 epochs, lr=5e-5

#### Option B: Command line

```bash
python main.py train \
    --model_name Miner \
    --pretrained_embedding distilroberta-base \
    --pretrained_tokenizer distilroberta-base \
    --user2id_path data/user2id.json \
    --category2id_path data/category2id.json \
    --train_behaviors_path data/MINDlarge_train/MINDlarge_train/behaviors.tsv \
    --train_news_path data/MINDlarge_train/MINDlarge_train/news.tsv \
    --eval_behaviors_path data/valid/behaviors.tsv \
    --eval_news_path data/valid/news.tsv \
    --max_title_length 20 \
    --max_sapo_length 64 \
    --his_length 50 \
    --apply_reduce_dim \
    --word_embed_dim 256 \
    --category_embed_dim 300 \
    --combine_type linear \
    --num_context_codes 32 \
    --context_code_dim 200 \
    --score_type weighted \
    --dropout 0.2 \
    --use_category_bias \
    --use_sapo \
    --npratio 4 \
    --train_batch_size 12 \
    --eval_batch_size 256 \
    --gradient_accumulation_steps 32 \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --max_grad_norm 0.8 \
    --weight_decay 0.01 \
    --fp16 \
    --fast_eval \
    --logging_steps 200 \
    --eval_steps 10000 \
    --metrics auc mrr ndcg@5 ndcg@10
```

#### Tham số quan trọng

| Tham số | Mặc định (config) | Mô tả |
|---------|---------|---------|
| `--use_sapo` | bật | Encode sapo cùng title |
| `--use_category_bias` | bật | Category embedding làm attention bias |
| `--apply_reduce_dim` | bật | Giảm chiều RoBERTa (768 → `word_embed_dim`) |
| `--word_embed_dim` | 256 | Chiều news embedding sau reduce |
| `--num_context_codes` | 32 | Số interest vectors K |
| `--score_type` | `weighted` | Cách tổng hợp K scores (`mean`/`max`/`weighted`) |
| `--combine_type` | `linear` | Kết hợp title+sapo (`linear`/`lstm`) |
| `--freeze_transformer` | tắt | Freeze weights DistilRoBERTa |
| `--resume_from_checkpoint` | - | Path checkpoint để resume/finetune |
| `--resume_training` | - | Restore cả optimizer/scheduler (nếu resume) |

### 2. Evaluation

```bash
python main.py eval @config/eval.txt

# hoặc command line
python main.py eval \
    --pretrained_embedding distilroberta-base \
    --pretrained_tokenizer distilroberta-base \
    --user2id_path data/user2id.json \
    --category2id_path data/category2id.json \
    --max_title_length 20 \
    --max_sapo_length 64 \
    --his_length 50 \
    --apply_reduce_dim \
    --word_embed_dim 256 \
    --category_embed_dim 300 \
    --combine_type linear \
    --num_context_codes 32 \
    --context_code_dim 200 \
    --score_type weighted \
    --dropout 0.2 \
    --use_category_bias \
    --use_sapo \
    --saved_model_path checkpoint/bestAucModel.pt \
    --data_name MIND \
    --eval_behaviors_path data/valid/behaviors.tsv \
    --eval_news_path data/valid/news.tsv \
    --eval_batch_size 512 \
    --fp16 \
    --metrics auc mrr ndcg@5 ndcg@10
```

### 3. Submission Generation

Tạo file prediction cho MIND test set (không có labels):

```bash
python main.py submission @config/submission.txt

# hoặc command line
python main.py submission \
    --pretrained_embedding distilroberta-base \
    --pretrained_tokenizer distilroberta-base \
    --user2id_path data/user2id.json \
    --category2id_path data/category2id.json \
    --max_title_length 20 \
    --max_sapo_length 64 \
    --his_length 50 \
    --apply_reduce_dim \
    --word_embed_dim 256 \
    --category_embed_dim 300 \
    --combine_type linear \
    --num_context_codes 32 \
    --context_code_dim 200 \
    --score_type weighted \
    --dropout 0.2 \
    --use_category_bias \
    --use_sapo \
    --saved_model_path checkpoint/bestAucModel.pt \
    --data_name MIND \
    --eval_behaviors_path data/MINDlarge_test/MINDlarge_test/behaviors.tsv \
    --eval_news_path data/MINDlarge_test/MINDlarge_test/news.tsv \
    --eval_batch_size 512 \
    --fp16
```

Output: `eval/{timestamp}/prediction.txt` (format: `impression_id [rank1,rank2,...]`)

## 🎭 Ensemble Methods

Dự án hỗ trợ 2 phương pháp ensemble để cải thiện hiệu suất:

### 1. Weighted Mean Ensemble

Kết hợp predictions bằng **trọng số tối ưu** (tìm bằng Bayesian Optimization với Optuna):

```bash
python phase2/run_ensemble.py \
    --predictions output_dir/pred_model1.txt \
                  output_dir/pred_model2.txt \
                  output_dir/pred_model3.txt \
    --truth phase2/ref/truth.txt \
    --method weighted \
    --output-dir phase2/ensemble_results \
    --n-trials 500
```

Output: `prediction_weighted_prod.txt` và `prediction_weighted_rank.txt`

### 2. Stacking Ensemble

Sử dụng **Logistic Regression** làm meta-model học cách kết hợp predictions:

```bash
python phase2/run_ensemble.py \
    --predictions output_dir/pred_model1.txt \
                  output_dir/pred_model2.txt \
                  output_dir/pred_model3.txt \
    --truth phase2/ref/truth.txt \
    --method stacking \
    --output-dir phase2/ensemble_results
```

Output: `prediction_stacking_prod.txt` và `prediction_stacking_rank.txt`

### Universal Inference (Production Mode)

Inference **không cần ground truth** (dùng weights/model đã train từ bước trên):

```bash
python phase2/universal_infer.py \
    --predictions model1_pred.txt model2_pred.txt model3_pred.txt \
    --weighted-dir phase2/ensemble_results/weighted_mean \
    --stacking-dir phase2/ensemble_results/stacking \
    --output-dir results \
    --methods all
```

### So sánh Ensemble Methods

| Method | Ưu điểm | Nhược điểm | Use case |
|--------|---------|------------|----------|
| **Weighted Mean** | Nhanh, dễ interpret | Chỉ linear combination | Models có AUC tương đồng |
| **Stacking** | Học non-linear patterns, robust hơn | Cần truth labels để train | Models đa dạng kiến trúc |

Xem thêm lý thuyết tại [phase2/ENSEMBLE_THEORY.md](phase2/ENSEMBLE_THEORY.md).

## 📈 Kết quả

Đánh giá trên **MIND Large dev set**:

### Base Models (MINER)

| Model | AUC | MRR | nDCG@5 | nDCG@10 |
|-------|-----|-----|--------|---------|
| MINER (no_sapo, member 1) | 0.6579 | 0.3201 | 0.3521 | 0.4135 |
| MINER (no_sapo, member 2) | 0.6653 | 0.3152 | 0.3468 | 0.4093 |
| MINER (with sapo, best) | **0.6793** | **0.3251** | **0.3585** | **0.4233** |

### Ensemble Results

| Method | AUC | MRR | nDCG@5 | nDCG@10 |
|--------|-----|-----|--------|---------|
| Weighted Mean | — | — | — | — |
| **Stacking** | **0.6890** | **0.3310** | **0.3629** | **0.4275** |

Stacking ensemble của 3 model cho kết quả tốt nhất (+0.0097 AUC so với best single model).

## 📁 Cấu trúc thư mục

```
MINER/
├── main.py                          # Entry point chính (train/eval/submission)
├── arguments.py                     # Định nghĩa tất cả CLI arguments
├── prepare_mind_mappings.py         # Tạo user2id.json, category2id.json
├── requirements.txt                 # Python dependencies
│
├── config/                          # Config files (dùng với @config/...)
│   ├── train.txt                   # Training config
│   ├── eval.txt                    # Evaluation config
│   └── submission.txt              # Submission config
│
├── src/                             # Source code MINER model
│   ├── model/
│   │   ├── model.py               # Miner: PolyAttention + TargetAwareAttention
│   │   └── news_encoder.py        # NewsEncoder (DistilRoBERTa-based)
│   ├── trainer.py                 # Training/eval/submission logic
│   ├── reader.py                  # MIND dataset reader & DataLoader
│   ├── evaluation.py              # AUC, MRR, nDCG metrics
│   ├── loss.py                    # Loss functions
│   └── utils.py                   # Utilities
│
├── data/                            # Dataset (không commit lên git)
│   ├── user2id.json
│   ├── category2id.json
│   ├── MINDlarge_train/
│   ├── MINDlarge_dev/
│   ├── MINDlarge_test/
│   ├── train/                      # Symlink hoặc copy cho train split
│   └── valid/                      # behaviors.tsv & news.tsv từ dev set
│
├── checkpoint/                      # Saved model checkpoints
│   ├── bestAucModel.pt             # MINER với sapo (AUC 0.6793)
│   ├── no_sapo/
│   │   ├── bestAucModel.pt        # MINER không sapo
│   │   ├── distillroberta_5e.pt
│   │   └── finalModel.pt
│   └── no_encoder_no_sapo/
│       └── finalModel.pt
│
├── fastformer-for-rec/              # Sub-module: SpeedyRec + Fastformer model
│   ├── train.py                    # Training SpeedyRec
│   ├── submission.py               # Inference SpeedyRec
│   ├── parameters.py               # Hyperparameters SpeedyRec
│   ├── models/                     # MLNR model (TextEncoder + Fastformer UserEncoder)
│   ├── data_handler/               # Streaming dataloader (TF-based)
│   └── NOTE.md                     # Giải thích chi tiết sub-module này
│
├── phase2/                          # Ensemble framework
│   ├── run_ensemble.py             # Script chạy ensemble chính
│   ├── ensemble.py                 # Weighted mean & stacking implementation
│   ├── universal_infer.py          # Production inference (không cần truth)
│   ├── optimize_hybrid_weights.py  # Tối ưu hybrid weights
│   ├── ENSEMBLE_THEORY.md          # Lý thuyết ensemble
│   ├── PRODUCTION_MODE_SUMMARY.md  # Hướng dẫn production
│   ├── ref/truth.txt               # Ground truth labels (dev set)
│   └── ensemble_results/           # Output predictions
│
├── output_dir/                      # Scores từ các model
│   ├── scores.txt                  # MINER best (AUC 0.6793)
│   ├── scores_stacking.txt         # Stacking ensemble (AUC 0.6890)
│   ├── scores_ngoc.txt             # Member 1 (AUC 0.6579)
│   └── scores_son.txt              # Member 2 (AUC 0.6653)
│
├── scripts/                         # Utility scripts
│   ├── sub_evaluator.py            # Evaluate predictions với truth file
│   ├── prod_to_rank.py             # Convert prod scores → rank format
│   └── generate_truth.py           # Tạo truth.txt từ behaviors.tsv
│
├── eval/                            # Evaluation outputs (auto-generated)
│   └── {timestamp}/
│       ├── args.json
│       └── prediction.txt
│
└── train/                           # Training checkpoints (auto-generated)
    └── {timestamp}/
```

## 📚 Tài liệu tham khảo

### Papers

1. **MINER**: Li et al. (2022) - [MINER: Multi-Interest Matching Network for News Recommendation](https://aclanthology.org/2022.findings-acl.29.pdf) — ACL 2022 Findings

2. **MIND Dataset**: Wu et al. (2020) - [MIND: A Large-scale Dataset for News Recommendation](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf) — ACL 2020

3. **Fastformer**: Wu et al. (2021) - [Fastformer: Additive Attention Can Be All You Need](https://arxiv.org/abs/2108.09084)

4. **SpeedyRec**: Liu et al. (2022) - [SpeedyRec: Efficient News Recommendation with News-history Knowledge Distillation](https://arxiv.org/abs/2205.04733)

