# MINER - Multi-Interest News Recommendation

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Triển khai mô hình **MINER (Multi-Interest Network for News Recommendation)** dựa trên paper [ACL 2022](https://aclanthology.org/2022.findings-acl.29.pdf), áp dụng cho bài toán gợi ý tin tức trên MIND dataset.

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
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Kết quả](#-kết-quả)
- [Tài liệu tham khảo](#-tài-liệu-tham-khảo)

## 🎯 Tổng quan

MINER là một mô hình neural network được thiết kế để:
- **Học multiple user interests** từ lịch sử đọc tin tức
- **Category-aware attention** để cải thiện khả năng đại diện tin tức
- **Multi-interest matching** giữa user history và candidate news
- **Ensemble predictions** từ nhiều models để tối ưu hiệu suất

### Tính năng chính

✅ **News Encoder**: Sử dụng pre-trained language model (RoBERTa/DistilRoBERTa) để encode title và sapo  
✅ **Category Embedding**: Tích hợp thông tin danh mục tin tức  
✅ **Multi-Head Attention**: Học K interests khác nhau của user  
✅ **Flexible Scoring**: Hỗ trợ nhiều phương pháp tổng hợp scores (mean, max, weighted)  
✅ **Ensemble Learning**: Weighted Mean & Stacking ensemble  
✅ **Production-ready**: Hỗ trợ inference mode không cần ground truth  

## 🏗️ Kiến trúc mô hình

```
┌─────────────────────────────────────────────────────────────┐
│                      MINER Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  News Encoder (Title + Sapo)                                │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │  RoBERTa     │ ──────> │  Linear      │                  │
│  │  Embedding   │         │  Projection  │                  │
│  └──────────────┘         └──────────────┘                  │
│         │                         │                          │
│         └─────────────┬───────────┘                          │
│                       ▼                                      │
│              Category Attention                              │
│              (Optional bias)                                 │
│                       │                                      │
│                       ▼                                      │
│            Multi-Interest User Encoder                       │
│         (K attention heads → K interests)                    │
│                       │                                      │
│                       ▼                                      │
│           Interest-Candidate Matching                        │
│          (Cosine similarity × K scores)                      │
│                       │                                      │
│                       ▼                                      │
│            Score Aggregation (mean/max/weighted)             │
│                       │                                      │
│                       ▼                                      │
│                 Click Probability                            │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Cài đặt

### Yêu cầu hệ thống

- Python >= 3.8
- CUDA (khuyến nghị cho training)
- RAM: >= 16GB
- Disk: >= 50GB (cho MIND dataset)

### Cài đặt dependencies

```bash
# Clone repository
git clone https://github.com/tranphuc15122004/MINER.git
cd MINER

# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### Dependencies chính

- `torch==2.1.0` - Deep learning framework
- `transformers==4.37.2` - Pre-trained language models
- `scikit-learn==1.3.2` - Machine learning utilities
- `pandas==2.1.3` - Data processing
- `tensorboard==2.15.1` - Training visualization

## 📊 Chuẩn bị dữ liệu

### 1. Download MIND dataset

```bash
# Download MINDlarge dataset
# Train set
wget https://mind201910.blob.core.windows.net/release/MINDlarge_train.zip
unzip MINDlarge_train.zip -d data/MINDlarge_train/

# Dev set
wget https://mind201910.blob.core.windows.net/release/MINDlarge_dev.zip
unzip MINDlarge_dev.zip -d data/MINDlarge_dev/

# Test set
wget https://mind201910.blob.core.windows.net/release/MINDlarge_test.zip
unzip MINDlarge_test.zip -d data/MINDlarge_test/
```

### 2. Chuẩn bị mappings

```bash
# Tạo user2id và category2id mappings
python prepare_mind_mappings.py \
    --train_behaviors data/MINDlarge_train/MINDlarge_train/behaviors.tsv \
    --train_news data/MINDlarge_train/MINDlarge_train/news.tsv \
    --output_dir data/
```

Kết quả tạo ra:
- `data/user2id.json` - Mapping user ID sang integer
- `data/category2id.json` - Mapping category sang integer

## 🚀 Sử dụng

### 1. Training

Sử dụng file config hoặc command line arguments:

#### Option A: Sử dụng config file

```bash
python main.py train @config/train.txt
```

#### Option B: Command line arguments

```bash
python main.py train \
    --model_name miner_base \
    --pretrained_embedding "distilroberta-base" \
    --pretrained_tokenizer "distilroberta-base" \
    --user2id_path data/user2id.json \
    --category2id_path data/category2id.json \
    --train_behaviors_path data/train/behaviors.tsv \
    --train_news_path data/train/news.tsv \
    --eval_behaviors_path data/valid/behaviors.tsv \
    --eval_news_path data/valid/news.tsv \
    --max_title_length 30 \
    --max_sapo_length 60 \
    --his_length 50 \
    --num_context_codes 20 \
    --context_code_dim 200 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --npratio 4 \
    --epochs 5 \
    --learning_rate 5e-5 \
    --use_category_bias \
    --use_sapo \
    --metrics auc mrr ndcg@5 ndcg@10
```

#### Tham số quan trọng:

- `--use_sapo`: Sử dụng cả sapo (abstract) ngoài title
- `--use_category_bias`: Kích hoạt category-aware attention
- `--freeze_transformer`: Freeze weights của pre-trained model
- `--apply_reduce_dim`: Giảm chiều embedding của RoBERTa
- `--num_context_codes`: Số lượng interests (K)
- `--score_type`: Cách tổng hợp scores (mean/max/weighted)

### 2. Evaluation

```bash
python main.py eval @config/eval.txt

# hoặc
python main.py eval \
    --saved_model_path checkpoint/bestAucModel.pt \
    --data_name valid \
    --eval_behaviors_path data/valid/behaviors.tsv \
    --eval_news_path data/valid/news.tsv \
    --eval_batch_size 128 \
    --metrics auc mrr ndcg@5 ndcg@10
```

### 3. Submission Generation

```bash
python main.py submission @config/submission.txt

# hoặc
python main.py submission \
    --saved_model_path checkpoint/bestAucModel.pt \
    --data_name test \
    --eval_behaviors_path data/MINDlarge_test/MINDlarge_test/behaviors.tsv \
    --eval_news_path data/MINDlarge_test/MINDlarge_test/news.tsv \
    --eval_batch_size 128
```

Output: File prediction tại `eval/{timestamp}/prediction.txt`

## 🎭 Ensemble Methods

Dự án hỗ trợ 2 phương pháp ensemble để cải thiện hiệu suất:

### 1. Weighted Mean Ensemble

Kết hợp predictions bằng **trọng số tối ưu** (tìm bằng Bayesian Optimization):

```bash
python phase2/run_ensemble.py \
    --predictions checkpoint/prediction_prod_Ngoc.txt \
                  checkpoint/prediction_prod_Phuc.txt \
                  checkpoint/prediction_prod_Son.txt \
    --truth ref/truth.txt \
    --method weighted \
    --output-dir phase2/ensemble_results \
    --n-trials 500
```

**Cách hoạt động:**
- Sử dụng Optuna để tìm trọng số tối ưu w₁, w₂, ..., wₙ
- Maximize impression-level AUC
- Output: `prediction_weighted_prod.txt` và `prediction_weighted_rank.txt`

### 2. Stacking Ensemble

Sử dụng **meta-model** (Logistic Regression) học cách kết hợp predictions:

```bash
python phase2/run_ensemble.py \
    --predictions checkpoint/prediction_prod_Ngoc.txt \
                  checkpoint/prediction_prod_Phuc.txt \
                  checkpoint/prediction_prod_Son.txt \
    --truth ref/truth.txt \
    --method stacking \
    --output-dir phase2/ensemble_results
```

**Cách hoạt động:**
- Train Logistic Regression trên predictions từ base models
- Sử dụng 5-fold cross-validation
- Output: `prediction_stacking_prod.txt` và `prediction_stacking_rank.txt`

### Universal Inference (Production Mode)

Cho production environment **không có ground truth**:

```bash
python phase2/universal_infer.py \
    --predictions model1_pred.txt model2_pred.txt model3_pred.txt \
    --weighted-dir phase2/ensemble_results/weighted_mean \
    --stacking-dir phase2/ensemble_results/stacking \
    --output-dir results \
    --methods all
```

**Tính năng:**
- ✅ Không bắt buộc truth file
- ✅ Tự động detect mode (evaluation vs inference)
- ✅ Hỗ trợ cả weighted và stacking
- ✅ Production-ready

### So sánh Ensemble Methods

| Method | Ưu điểm | Nhược điểm | Use case |
|--------|---------|------------|----------|
| **Weighted Mean** | ⚡ Nhanh, đơn giản<br>📊 Dễ interpret weights | 🔢 Chỉ linear combination | Models tương đồng nhau |
| **Stacking** | 🎯 Học nonlinear patterns<br>💪 Robust hơn | ⏱️ Chậm hơn<br>🎓 Cần thêm data | Models đa dạng |

Xem thêm chi tiết tại [ENSEMBLE_THEORY.md](phase2/ENSEMBLE_THEORY.md).

## 📁 Cấu trúc thư mục

```
MINER/
├── main.py                      # Entry point chính
├── arguments.py                 # Định nghĩa arguments
├── requirements.txt             # Dependencies
│
├── config/                      # Config files
│   ├── train.txt               # Training config
│   ├── eval.txt                # Evaluation config
│   └── submission.txt          # Submission config
│
├── src/                        # Source code
│   ├── model/
│   │   ├── model.py           # MINER model
│   │   └── news_encoder.py   # News encoder
│   ├── trainer.py             # Training logic
│   ├── reader.py              # Data loading
│   ├── evaluation.py          # Metrics
│   └── utils.py               # Utilities
│
├── data/                       # Dataset
│   ├── user2id.json
│   ├── category2id.json
│   ├── MINDlarge_train/
│   ├── MINDlarge_dev/
│   └── MINDlarge_test/
│
├── checkpoint/                 # Saved models
│   ├── bestAucModel.pt
│   └── finalModel.pt
│
├── phase2/                     # Ensemble methods
│   ├── ensemble.py            # Ensemble implementation
│   ├── run_ensemble.py        # Run ensemble
│   ├── universal_infer.py     # Production inference
│   ├── ENSEMBLE_THEORY.md     # Ensemble theory
│   └── ensemble_results/      # Ensemble outputs
│
├── scripts/                    # Utility scripts
│   ├── sub_evaluator.py       # Evaluate predictions
│   ├── prod_to_rank.py        # Convert prod → rank
│   └── generate_truth.py      # Generate truth file
│
└── train/                      # Training logs
    └── {timestamp}/
        ├── args.json
        └── *.pt
```

## 📚 Tài liệu tham khảo

### Papers

1. **MINER**: Li et al. (2022) - [Efficiently Leveraging Multi-level User Intent for Session-based Recommendation via Atten-Mixer Network](https://aclanthology.org/2022.findings-acl.29.pdf)

2. **MIND Dataset**: Wu et al. (2020) - [MIND: A Large-scale Dataset for News Recommendation](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf)

