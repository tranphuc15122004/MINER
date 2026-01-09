# Lý Thuyết Ensemble Methods: Weighted Mean & Stacking

Tài liệu này giải thích chi tiết về 2 phương pháp ensemble được sử dụng trong hệ thống recommendation, dựa trên implementation của team sugawarya (RecSys Challenge 2024 Winner).

---

## 1. WEIGHTED MEAN ENSEMBLE

### 1.1. Ý tưởng cơ bản

Weighted Mean Ensemble kết hợp predictions từ nhiều models bằng cách tính **trung bình có trọng số**:

```
P_ensemble = w₁ × P₁ + w₂ × P₂ + ... + wₙ × Pₙ
```

Trong đó:
- `Pᵢ`: Prediction từ model thứ i
- `wᵢ`: Trọng số của model i (0 ≤ wᵢ ≤ 1)
- `P_ensemble`: Prediction cuối cùng

### 1.2. Vấn đề: Tìm trọng số tối ưu

**Câu hỏi:** Làm thế nào để tìm bộ trọng số {w₁, w₂, ..., wₙ} tốt nhất?

**Giải pháp:** Sử dụng **Bayesian Optimization** (Optuna library)

### 1.3. Quy trình tối ưu trọng số

#### Bước 1: Data Sampling
```
Validation set → Sample 1/100 impressions → Mini validation set
```
**Lý do:** Giảm thời gian tính toán trong quá trình tối ưu

#### Bước 2: Define Objective Function
```python
objective(w₁, w₂, ..., wₙ):
    # Tính prediction ensemble
    P_ensemble = w₁×P₁ + w₂×P₂ + ... + wₙ×Pₙ
    
    # Tính AUC cho từng impression
    for each impression:
        auc_i = AUC(y_true_i, P_ensemble_i)
    
    # Return mean AUC
    return mean(auc₁, auc₂, ..., aucₘ)
```

**Impression-level AUC:**
- Tính AUC riêng cho TỪNG impression (group of candidates)
- Mean AUC = average của tất cả impression AUCs
- Phù hợp với cách đánh giá của MIND dataset

#### Bước 3: Bayesian Optimization
```
Optuna trials:
  Trial 1: Try w₁=0.3, w₂=0.7 → AUC = 0.65
  Trial 2: Try w₁=0.5, w₂=0.5 → AUC = 0.67
  Trial 3: Try w₁=0.6, w₂=0.4 → AUC = 0.68
  ...
  Trial 200: Try w₁=0.58, w₂=0.42 → AUC = 0.69 (best)
```

**Optuna advantages:**
- Thông minh hơn Grid Search (không thử tất cả combinations)
- Học từ trials trước để suggest trials mới tốt hơn
- Converge nhanh hơn Random Search

### 1.4. Prediction với weights đã tối ưu

```
For new data:
  P_ensemble = 0.58×P₁ + 0.42×P₂
```

### 1.5. Ưu & Nhược điểm

**Ưu điểm:**
- ✅ Đơn giản, dễ hiểu
- ✅ Nhanh (chỉ weighted sum)
- ✅ Robust (ít overfitting)
- ✅ Interpretable (biết model nào quan trọng hơn)

**Nhược điểm:**
- ❌ Linear combination only (không học được non-linear relationships)
- ❌ Không tận dụng được interaction giữa predictions
- ❌ Tất cả impressions dùng cùng weights (không adaptive)

---

## 2. STACKING ENSEMBLE

### 2.1. Ý tưởng cơ bản

Stacking sử dụng một **meta-model** (LightGBM) để học cách kết hợp predictions:

```
Base predictions → Feature Engineering → Meta-model → Final prediction
```

**Khác với Weighted Mean:**
- Weighted: P = w₁P₁ + w₂P₂ (linear)
- Stacking: P = f(P₁, P₂, features) (non-linear, phức tạp hơn)

### 2.2. Feature Engineering

Không chỉ dùng raw predictions [P₁, P₂], mà **tạo thêm nhiều features** từ chúng:

#### 2.2.1. Statistical Features (trong impression)
```
Cho mỗi prediction P:
  - P_mean: Mean của P trong impression
  - P_max: Max của P trong impression  
  - P_min: Min của P trong impression
  - P_std: Standard deviation của P trong impression
```

**Ví dụ:** Impression có 5 candidates với P₁ = [0.8, 0.3, 0.6, 0.2, 0.9]
```
P₁_mean = 0.56
P₁_max = 0.9
P₁_min = 0.2
P₁_std = 0.28
```

#### 2.2.2. Normalized Features
```
P_zscore = (P - P_mean) / (P_std + ε)
P_normed = (P - P_min) / (P_max - P_min + ε)
```

**Ý nghĩa:**
- Z-score: Bao nhiêu standard deviations từ mean?
- Normalized: Vị trí tương đối trong [0,1]

#### 2.2.3. Rank Features
```
P_rank = Rank of P within impression (ascending)
P_rank_desc = Rank of P within impression (descending)
P_normedrank = P_rank / impression_size
P_normedrank_desc = P_rank_desc / impression_size
```

**Ví dụ:** P = [0.8, 0.3, 0.6, 0.2, 0.9]
```
P_rank_desc = [2, 4, 3, 5, 1]
P_normedrank_desc = [0.4, 0.8, 0.6, 1.0, 0.2]
```

#### 2.2.4. Pairwise Features (giữa 2 models)
```
diff = P₁ - P₂
ratio = P₁ / (P₂ + ε)
max = max(P₁, P₂)
min = min(P₁, P₂)
```

**Ý nghĩa:** Capture relationships giữa predictions của 2 models

#### 2.2.5. Aggregate Features
```
pred_mean_all = Mean của tất cả predictions
impression_count = Số candidates trong impression
```

**Tổng kết features:**
- 2 base predictions → ~100+ engineered features
- Mỗi feature capture một khía cạnh khác nhau của data

### 2.3. Meta-model: LightGBM với LambdaRank

#### Tại sao LightGBM?
- 🎯 Gradient Boosting: Mạnh với tabular data
- 🎯 LambdaRank objective: Được thiết kế CHO RANKING tasks
- 🎯 Nhanh, efficient với nhiều features

#### LambdaRank Objective

**Ranking problem:**
- Không chỉ predict label (0/1)
- Mà predict **thứ tự** đúng của candidates

**LambdaRank:**
- Tối ưu hóa **ranking metrics** (NDCG@k) trực tiếp
- Học cách sắp xếp candidates đúng thứ tự
- Tính gradient dựa trên pairwise comparisons

**NDCG@k (Normalized Discounted Cumulative Gain):**
```
DCG@k = Σᵢ₌₁ᵏ (2^relᵢ - 1) / log₂(i+1)
NDCG@k = DCG@k / IDCG@k
```
- Đánh giá cao việc rank đúng items quan trọng (clicked)
- Giảm trọng số theo position (position 1 > position 10)

#### LightGBM Parameters
```python
lgb_params = {
    'objective': 'lambdarank',      # Ranking objective
    'metric': 'ndcg',                # Optimize NDCG
    'ndcg_at': [5, 10],             # Evaluate at top-5, top-10
    'learning_rate': 0.1,            # Learning rate
    'feature_fraction': 0.8,         # Use 80% features per tree
    'bagging_fraction': 0.8,         # Use 80% samples per iteration
    'bagging_freq': 1,               # Bagging every iteration
    'max_bin': 1024,                 # Max bins for features
}
```

### 2.4. Cross-Validation: GroupKFold

**Vấn đề:** Candidates trong cùng impression có correlation

**Giải pháp:** GroupKFold
```
Fold 1: Train on impressions [1000-5000], Valid on [0-1000]
Fold 2: Train on [0-1000, 2000-5000], Valid on [1000-2000]
...
Fold k: Train on other folds, Valid on this fold
```

**Quan trọng:**
- Tất cả candidates của 1 impression phải trong CÙNG fold
- Tránh data leakage giữa train/valid

### 2.5. Out-of-Fold (OOF) Predictions

```
Training flow:
  Fold 1: Train model₁, predict on valid_fold₁ → oof_pred₁
  Fold 2: Train model₂, predict on valid_fold₂ → oof_pred₂
  ...
  Fold k: Train modelₖ, predict on valid_foldₖ → oof_predₖ
  
  Combine: oof_predictions = [oof_pred₁, oof_pred₂, ..., oof_predₖ]
  Calculate: OOF_AUC = AUC(y_true, oof_predictions)
```

**OOF AUC:**
- Đánh giá **unbiased** performance
- Mỗi sample được predict bởi model CHƯA thấy nó trong training
- Gần với test performance hơn train performance

### 2.6. Inference (Prediction)

```
For new data:
  1. Create features (same engineering)
  2. Predict với MỖI fold model:
     pred₁ = model₁.predict(features)
     pred₂ = model₂.predict(features)
     ...
     predₖ = modelₖ.predict(features)
  3. Average predictions:
     final_pred = (pred₁ + pred₂ + ... + predₖ) / k
```

**Averaging k models:**
- Giảm variance
- Robust hơn single model
- Exploit diversity giữa các folds

### 2.7. Ưu & Nhược điểm

**Ưu điểm:**
- ✅ Non-linear combination (học relationships phức tạp)
- ✅ Feature engineering → capture nhiều patterns
- ✅ LambdaRank → optimize directly cho ranking
- ✅ Thường performance cao hơn Weighted Mean

**Nhược điểm:**
- ❌ Phức tạp hơn (nhiều bước, nhiều hyperparameters)
- ❌ Chậm hơn (training + feature engineering)
- ❌ Dễ overfit hơn (cần careful tuning)
- ❌ Ít interpretable (black-box model)

---

## 3. SO SÁNH WEIGHTED MEAN VS STACKING

| Tiêu chí | Weighted Mean | Stacking |
|----------|---------------|----------|
| **Complexity** | Simple (weighted sum) | Complex (feature eng. + model) |
| **Speed** | Rất nhanh | Chậm hơn |
| **Performance** | Tốt | Thường tốt hơn |
| **Interpretability** | Cao (xem weights) | Thấp (black-box) |
| **Overfitting risk** | Thấp | Trung bình-Cao |
| **Training time** | Nhanh (~minutes) | Chậm (~hours) |
| **Inference time** | Rất nhanh | Nhanh |

---

## 4. KHI NÀO DÙNG PHƯƠNG PHÁP NÀO?

### Dùng Weighted Mean khi:
- ✅ Cần solution đơn giản, nhanh
- ✅ Models tương đối independent
- ✅ Muốn interpretability cao
- ✅ Ít thời gian training
- ✅ Dataset nhỏ (dễ overfit với stacking)

### Dùng Stacking khi:
- ✅ Cần squeeze maximum performance
- ✅ Có đủ data (tránh overfit)
- ✅ Có time/resource để train meta-model
- ✅ Models có complementary strengths
- ✅ Task phức tạp (non-linear relationships)

### Best Practice: Thử CẢ HAI
```
1. Start với Weighted Mean (baseline)
2. Implement Stacking (push performance)
3. Compare results
4. Choose based on requirements (speed vs accuracy)
```

---

## 5. LƯU Ý QUAN TRỌNG

### 5.1. Data Leakage
- ❌ KHÔNG train meta-model trên cùng data dùng để train base models
- ✅ Dùng separate validation set hoặc OOF predictions
- ✅ GroupKFold để tránh leak giữa candidates trong impression

### 5.2. Overfitting
- ⚠️ Stacking dễ overfit nếu không careful
- ✅ Monitor OOF AUC vs validation AUC
- ✅ Regularization (max_depth, min_samples, etc.)
- ✅ Early stopping

### 5.3. Diversity của Base Models
- 💡 Ensemble hoạt động tốt khi base models **diverse**
- 💡 Models quá giống nhau → ensemble không giúp nhiều
- 💡 Check correlation matrix giữa predictions

### 5.4. Computational Cost
- ⏱️ Weighted: O(n) - linear với data size
- ⏱️ Stacking: O(n × k × trees) - phụ thuộc nhiều factors
- 💡 Trade-off giữa accuracy gain vs computational cost

---

## 6. TOÁN HỌC CHI TIẾT

### 6.1. Weighted Mean Optimization

**Problem:**
```
maximize: E_impressions[AUC(y_true, Σᵢ wᵢPᵢ)]
subject to: 0 ≤ wᵢ ≤ 1
```

**Optuna sử dụng:**
- Tree-structured Parzen Estimator (TPE)
- Mô hình hóa P(score|weights) và P(weights)
- Chọn weights maximize P(weights|score > threshold)

### 6.2. LambdaRank Gradient

**Pairwise ranking:**
```
For pair (i, j) where relᵢ > relⱼ:
  λᵢⱼ = -∂C / ∂sᵢ
  
  Where C = cost function based on ranking metric
  sᵢ = model score for item i
```

**Update:**
```
sᵢ ← sᵢ + η × Σⱼ λᵢⱼ
```

**NDCG gradient:**
- Tính impact nếu swap positions i, j
- Gradient ∝ |ΔNDCG| × sigmoid(sᵢ - sⱼ)

---

## 7. KẾT LUẬN

**Weighted Mean Ensemble:**
- Phương pháp đơn giản nhưng hiệu quả
- Tối ưu weights bằng Bayesian Optimization
- Phù hợp làm baseline và production nhanh

**Stacking Ensemble:**
- Phương pháp mạnh mẽ với feature engineering
- Meta-model (LightGBM + LambdaRank) học non-linear combinations
- Thường cho performance tốt nhất nhưng phức tạp hơn

**Recommendation:**
- Development: Implement CẢ HAI
- Production: Choose dựa trên trade-off accuracy vs latency
- Best practice: Weighted cho speed, Stacking cho accuracy
