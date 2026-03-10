# Ghi chú: Fastformer-for-Rec trong Project MINER

## 1. Tổng quan

`fastformer-for-rec/` là một **sub-module độc lập** implement mô hình **SpeedyRec + Fastformer** cho bài toán **News Recommendation** trên dataset **MIND (Microsoft News Dataset)**. Đây là phiên bản unofficial của bài báo Fastformer, kết hợp avec PLM (Pre-trained Language Model) làm news encoder.

---

## 2. Kiến trúc mô hình

### 2.1 Model chính: `MLNR` (`models/speedyrec.py`)

```
MLNR
├── news_encoder  → TextEncoder   (encode bài báo → vector)
└── user_encoder  → UserEncoder   (encode lịch sử đọc của user → vector)
```

**Forward pass:**
1. Lấy vector các bài báo candidate và lịch sử đọc của user từ `news_vecs` (cache hoặc encode mới).
2. Tính `user_vec` từ lịch sử đọc qua `UserEncoder`.
3. Tính score = dot-product(`candidate_vec`, `user_vec`).
4. Loss = CrossEntropy(score, labels).

### 2.2 `TextEncoder` (`models/speedyrec.py`)

- Dùng PLM (`TuringNLRv3` / `BertModel` / `AutoModel`) để encode title (và abstract nếu có).
- PLM encode từng câu → mean pooling (hoặc attention pooling nếu có abstract).
- Linear projection → `news_dim` (mặc định 256).
- Nếu dùng cả title + abstract: kết hợp qua `AttentionPooling`.

**Cấu hình `MODEL_CLASSES` trong `utility/utils.py`:**
```python
'unilm'  → TuringNLRv3ForSequenceClassification + TuringNLRv3Tokenizer
'others' → AutoModel + AutoTokenizer  (roberta-base, deberta-base, ...)
```

### 2.3 `UserEncoder` (`models/speedyrec.py`)

- Nhận sequence các news vectors (lịch sử đọc của user, tối đa `user_log_length=100`).
- Đưa qua **Fastformer** (`models/fast.py`): multi-head self-attention encoder.
- Output: pooled user vector kích thước `news_dim`.

### 2.4 `Fastformer` (`models/fast.py`)

Đây là custom implementation của Fastformer encoder:
- `FastAttention`: Multi-head self-attention với scaled dot-product.
- `FastformerLayer`: Self-attn + FFN + residual + LayerNorm (giống Transformer layer chuẩn).
- `FastformerEncoder`: Stack nhiều `FastformerLayer`.
- `Fastformer` (wrapper): encode sequence → mean pooling có mask → output vector.

Cấu hình đọc từ `models/ffconfig.json`:
- `num_hidden_layers`: số layer (mặc định 2)
- `num_attention_heads`: số head (mặc định 8)
- `hidden_dropout_prob`: dropout rate

---

## 3. Cấu trúc thư mục

```
fastformer-for-rec/
├── train.py              # Entry point huấn luyện (chạy trực tiếp)
├── submission.py         # Entry point tạo file dự đoán để nộp
├── parameters.py         # Toàn bộ hyperparameter (parse_args)
├── data_generation.py    # Tạo dữ liệu SpeedyRec format từ MIND raw
├── models/
│   ├── speedyrec.py      # MLNR model (TextEncoder + UserEncoder)
│   ├── fast.py           # Fastformer implementation
│   ├── ffconfig.json     # Config cho Fastformer user encoder
│   └── tnlrv3/           # TuringNLRv3 (UniLM v2) model code
├── data_handler/
│   ├── preprocess.py     # Tokenize news, build news feature arrays
│   ├── streaming.py      # TF-based streaming data reader
│   ├── TrainDataloader.py# DataLoader train với news cache
│   └── TestDataloader.py # DataLoader eval/test
└── utility/
    ├── utils.py          # logger, device, lr_schedule, MODEL_CLASSES
    └── metrics.py        # AUC, MRR, nDCG@5, nDCG@10
```

---

## 4. Pipeline huấn luyện (`train.py`)

### Flow chính:

```
ddp_train_vd(args)
    └── train(local_rank=0, ...)
            ├── get_news_feature(args, mode='train')  → news_info, news_combined
            ├── MLNR(args) + Adam optimizer (2 param groups)
            │       group[0]: PLM params   → lr = pretrain_lr (8e-6)
            │       group[1]: rest params  → lr = lr (1e-4)
            ├── (optional) _load_checkpoint(ckpt_path) để resume
            └── for ep in range(epochs):
                    DataLoaderTrainForSpeedyRec(...)  ← streaming + cache
                    for batch in dataloader:
                        ├── encode_vecs = news_encoder(input_ids)      [encode mới]
                        ├── cache_vec = cache[address_cache]            [từ cache]
                        ├── news_vecs = cat(cache_vec, encode_vecs)
                        ├── bz_loss, y_hat = MLNR(news_vecs, ...)
                        ├── loss.backward() + optimizer.step()
                        ├── update cache[update_cache] = encode_vecs
                        └── (mỗi save_steps) → lưu checkpoint .pt
                    └── test(model, ...) sau mỗi epoch → AUC
```

### Cơ chế **News Cache** (đặc trưng của SpeedyRec):
- `cache = np.zeros((num_news, news_dim))` - cache toàn bộ news vectors.
- Mỗi batch: một số news lấy từ cache (hit), số còn lại encode mới.
- Sau khi encode, cập nhật cache. Tỉ lệ cache hit tăng dần.
- Hyperparameter: `beta_for_cache` (tốc độ tăng lookup probability), `max_step_in_cache` (γ).

---

## 5. Hyperparameters quan trọng (`parameters.py`)

| Tham số | Mặc định | Ý nghĩa |
|---------|---------|---------|
| `news_dim` | 64 (project dùng 256) | Kích thước news/user vector |
| `num_hidden_layers` | -1 (full) | Số layer PLM (dùng 8 để giảm) |
| `user_log_length` | 100 | Số bài báo lịch sử tối đa |
| `num_words_title` | 32 | Số token title |
| `num_words_abstract` | 50 | Số token abstract |
| `npratio` | 1 | Số negative sample / positive |
| `pretrain_lr` | 1e-4 | LR cho PLM layers |
| `lr` | 1e-4 | LR cho các layer còn lại |
| `batch_size` | 64 | Batch size |
| `beta_for_cache` | 0.002 | Cache lookup growth rate |
| `max_step_in_cache` | 20 | Max steps news stays in cache |
| `news_attributes` | `['title', 'abstract']` | Thuộc tính news dùng để encode |

---

## 6. Cách sử dụng trong Project MINER

### 6.1 Vị trí trong pipeline

```
MINER project
├── src/           ← MINER model chính (NRMS-based, dùng DistilRoBERTa)
├── fastformer-for-rec/  ← Sub-model SpeedyRec (dùng UniLM/BERT + Fastformer)
│       └── checkpoint: checkpoint/no_sapo/*, checkpoint/bestAucModel.pt
└── phase2/        ← Ensemble các model (MINER + SpeedyRec + ...)
```

### 6.2 Checkpoint đã train

Các checkpoint của fastformer-for-rec nằm tại:
- `checkpoint/bestAucModel.pt` - checkpoint tốt nhất theo AUC
- `checkpoint/no_sapo/finalModel.pt` - model train không dùng abstract
- `checkpoint/no_sapo/bestAucModel.pt`, `distillroberta_5e.pt`, `distilRoberta_3e.pt`

### 6.3 Chạy inference (dùng checkpoint sẵn)

```bash
cd fastformer-for-rec

python submission.py \
  --pretreained_model others \
  --pretrained_model_path distilbert-base-uncased \
  --root_data_dir ../data/speedy_data/ \
  --num_hidden_layers 8 \
  --load_ckpt_name ../checkpoint/no_sapo/finalModel.pt \
  --batch_size 256 \
  --news_attributes title \
  --news_dim 256
```

Hoặc dùng UniLM (nếu có checkpoint UniLM):
```bash
python submission.py \
  --pretreained_model unilm \
  --pretrained_model_path ./speedymind_ckpts \
  --root_data_dir ../data/speedy_data/ \
  --num_hidden_layers 8 \
  --load_ckpt_name ../checkpoint/bestAucModel.pt \
  --batch_size 256 \
  --news_attributes title \
  --news_dim 256
```

### 6.4 Resume training

```bash
cd fastformer-for-rec

python train.py \
  --pretreained_model others \
  --pretrained_model_path distilbert-base-uncased \
  --root_data_dir ../data/speedy_data/ \
  --news_attributes title \
  --num_hidden_layers 8 \
  --lr 1e-4 \
  --pretrain_lr 8e-6 \
  --warmup True \
  --schedule_step 240000 \
  --warmup_step 1000 \
  --batch_size 64 \
  --npratio 4 \
  --news_dim 256 \
  --load_ckpt_name ../checkpoint/no_sapo/finalModel.pt \
  --savename speedyrec_finetune
```

### 6.5 Tích hợp vào Ensemble (phase2/)

File dự đoán của fastformer-for-rec (định dạng `impression_id [rank1,rank2,...]`) được dùng trong:
- `phase2/ensemble.py` - weighted/stacking ensemble combining nhiều model
- `phase2/run_ensemble.py` - script chạy ensemble
- `output_dir/scores_*.txt` - scores từ các model khác nhau

Để dùng prediction của fastformer trong ensemble:
1. Chạy `submission.py` → `prediction.zip` → giải nén ra `prediction.txt`
2. Copy vào `output_dir/` hoặc `phase2/ensemble_results/`
3. Chạy `phase2/run_ensemble.py` để combine

---

## 7. Lưu ý kỹ thuật

### 7.1 Dependency đặc biệt
- **TensorFlow** được dùng trong `data_handler/streaming.py` để đọc data files qua `tf.data`. Đây là dependency nặng - cần install TF ngay cả khi train trên PyTorch.
- **TuringNLRv3** (`models/tnlrv3/`): custom UniLM v2 implementation. Cần download checkpoint riêng nếu dùng `--pretreained_model unilm`.
- Nếu dùng `--pretreained_model others`: dùng bất kỳ HuggingFace model nào (distilbert, roberta, ...).

### 7.2 Format data đầu vào
Data phải qua `data_generation.py` để convert từ MIND raw format sang SpeedyRec format (`.tsv` files trong `data/speedy_data/train/`, `data/speedy_data/dev/`).

### 7.3 Checkpoint format
```python
{
    'model_state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),   # (không load khi resume để tránh lỗi)
    'category_dict': {...},
    'subcategory_dict': {...},
}
```

### 7.4 Single GPU mode
`train.py` đã được modify để chạy **single process** (không DDP), dùng `mp.Manager` cho shared state nhưng không dùng `dist.init_process_group`. Hàm `ddp_train_vd` ép `world_size=1`.

### 7.5 Metrics
Eval dùng 4 metrics tiêu chuẩn của MIND:
- **AUC**: Area Under ROC Curve
- **MRR**: Mean Reciprocal Rank  
- **nDCG@5 / nDCG@10**: Normalized Discounted Cumulative Gain

---

