# Cải tiến hàm eval() để lưu kết quả prediction

## Tóm tắt thay đổi

Đã cải tiến hàm `eval()` trong file [src/trainer.py](src/trainer.py) để vừa đánh giá model vừa lưu kết quả prediction theo định dạng CodaLab.

## Chi tiết triển khai

### 1. Thay đổi trong hàm `_eval()` (dòng 476-591)

**Thêm thu thập predictions:**
- Khởi tạo dict `impression_predictions` khi `save_result=True`
- Trong vòng lặp batch, thu thập predictions với cấu trúc:
  ```python
  impression_predictions[imp_id] = [(news_id, score), ...]
  ```
- Sau khi đánh giá xong, gọi `_save_prediction_file()` để lưu file

**Code thêm vào:**
```python
# For saving prediction results
impression_predictions = {} if save_result else None

# ... trong vòng lặp batch ...
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

# ... sau khi tính scores ...
if save_result and impression_predictions:
    self._save_prediction_file(impression_predictions)
```

### 2. Hàm mới `_save_prediction_file()` (dòng 600-632)

Hàm này chuyển đổi predictions thành định dạng CodaLab:

**Định dạng output:**
```
ImpressionID [Rank-of-News1,Rank-of-News2,...,Rank-of-NewsN]
```

**Ví dụ:**
```
24481 [4,1,3,2]
```

**Logic ranking:**
1. Sắp xếp candidate news theo score giảm dần
2. Gán rank: 1 (cao nhất), 2, 3, ... (thấp nhất)
3. Tạo list ranks theo thứ tự vị trí ban đầu của candidates

**Code:**
```python
def _save_prediction_file(self, impression_predictions):
    """
    Save prediction results in CodaLab submission format.
    
    Format: ImpressionID [Rank-of-News1,Rank-of-News2,...,Rank-of-NewsN]
    Where ranks indicate the ranking order (1=best, higher=worse)
    """
    output_path = os.path.join(self._path, 'prediction.txt')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for imp_id in sorted(impression_predictions.keys()):
            news_scores = impression_predictions[imp_id]
            
            # Sort by score (descending) to get ranking
            sorted_indices = sorted(range(len(news_scores)), 
                                  key=lambda i: news_scores[i][1], 
                                  reverse=True)
            
            # Create rank list
            ranks = [0] * len(sorted_indices)
            for rank, idx in enumerate(sorted_indices, start=1):
                ranks[idx] = rank
            
            # Write to file
            ranks_str = ','.join(map(str, ranks))
            f.write(f'{imp_id} [{ranks_str}]\n')
    
    self._logger.info(f'Prediction file saved to: {output_path}')
    self._logger.info(f'Total impressions: {len(impression_predictions)}')
```

## Cách sử dụng

Khi chạy evaluation với `save_eval_result=True`, file `prediction.txt` sẽ được tạo tự động trong thư mục output:

```bash
python main.py --mode eval --save_eval_result
```

File output sẽ có tại: `eval/[timestamp]/prediction.txt`

## Ví dụ minh họa

Đã tạo file test [test_prediction_format.py](test_prediction_format.py) để minh họa logic:

```
Impression ID: 24481
Candidate News: N125045 N87192 N73556 N20417
Scores:         0.3     0.9    0.5    0.7

Output: 24481 [4,1,3,2]

Giải thích:
- N125045 (vị trí 0, score 0.3) -> Rank 4 (thấp nhất)
- N87192  (vị trí 1, score 0.9) -> Rank 1 (cao nhất)
- N73556  (vị trí 2, score 0.5) -> Rank 3
- N20417  (vị trí 3, score 0.7) -> Rank 2
```

## Lợi ích

1. ✅ Tích hợp trong quá trình đánh giá - không cần chạy riêng
2. ✅ Định dạng chuẩn CodaLab - ready để submit
3. ✅ Tự động sắp xếp và tính rank
4. ✅ Không ảnh hưởng đến logic đánh giá hiện tại
5. ✅ Chỉ chạy khi cần (save_result=True)
