"""
Ensemble methods based on sugawarya (RecSys Challenge 2024 winner)

Phương pháp:
1. Weighted Mean: Tối ưu trọng số bằng Optuna trên validation set
2. LightGBM Stacking: Feature engineering + LambdaRank
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import optuna
from tqdm import tqdm
import lightgbm as lgb
import json
import pickle
import os
from datetime import datetime



def get_prod_predict(path: str = 'checkpoint/prediction_prod.txt'):
    """Parse a prediction file of the form:
    <impression_id> [score1,score2,...]

    Returns a dict mapping impression_id (int) -> list[float].
    """
    import ast
    results = {}
    if path is None:
        return results

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # split on first whitespace
            try:
                impid_str, arr_str = line.split(None, 1)
            except ValueError:
                # malformed line
                continue
            try:
                scores = ast.literal_eval(arr_str)
            except Exception:
                # fallback: try to strip surrounding brackets
                s = arr_str.strip()
                if s.startswith('[') and s.endswith(']'):
                    s = s[1:-1]
                parts = [p.strip() for p in s.split(',') if p.strip()]
                try:
                    scores = [float(x) for x in parts]
                except Exception:
                    continue
            # convert impression id to int when possible
            try:
                impid = int(impid_str)
            except Exception:
                impid = impid_str
            results[impid] = [float(x) for x in scores]

    return results


def load_truth(path: str = 'phase2/ref/truth.txt'):
    """Load truth file where each line is:
    <impression_id> <json_array_of_labels>

    Returns dict mapping impression_id (int) -> list[int].
    """
    import json
    results = {}
    if path is None:
        return results

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                impid_str, labels_str = s.split(None, 1)
            except ValueError:
                continue
            try:
                labels = json.loads(labels_str)
            except Exception:
                # fallback: try to parse a simple bracket list
                ls = labels_str.strip()
                if ls.startswith('[') and ls.endswith(']'):
                    ls = ls[1:-1]
                parts = [p.strip() for p in ls.split(',') if p.strip()]
                try:
                    labels = [int(x) for x in parts]
                except Exception:
                    continue
            try:
                impid = int(impid_str)
            except Exception:
                impid = impid_str
            results[impid] = [int(x) for x in labels]

    return results


def load_predictions_as_df(pred_paths: List[str], truth_path: str = None) -> pd.DataFrame:
    """
    Load multiple prediction files và (optional) truth file thành DataFrame
    
    Args:
        pred_paths: List đường dẫn đến các file prediction
        truth_path: Đường dẫn đến file truth (optional - chỉ cần khi training/evaluating)
        
    Returns:
        DataFrame với columns: impression_id, candidate_idx, [target], pred_0, pred_1, ...
        (target column chỉ có khi truth_path được cung cấp)
    """    
    # Load predictions
    pred_dicts = [get_prod_predict(path) for path in pred_paths]
    
    # Load truth if provided
    truth_dict = load_truth(truth_path) if truth_path else None
    
    # Build DataFrame
    rows = []
    
    if truth_dict is not None:
        # Training/evaluation mode: iterate based on truth
        for imp_id, labels in tqdm(truth_dict.items(), desc="Building DataFrame"):
            # Bỏ qua impression rỗng
            if len(labels) == 0:
                continue
                
            for cand_idx, label in enumerate(labels):
                row = {
                    'impression_id': imp_id,
                    'candidate_idx': cand_idx,
                    'target': label
                }
                
                # Add predictions from each model
                for model_idx, pred_dict in enumerate(pred_dicts):
                    if imp_id in pred_dict and cand_idx < len(pred_dict[imp_id]):
                        row[f'pred_{model_idx}'] = pred_dict[imp_id][cand_idx]
                    else:
                        row[f'pred_{model_idx}'] = 0.0  # Default score
                
                rows.append(row)
    else:
        # Inference mode: iterate based on predictions only
        # Use first prediction dict as reference for impression IDs
        for imp_id, scores in tqdm(pred_dicts[0].items(), desc="Building DataFrame"):
            if len(scores) == 0:
                continue
                
            for cand_idx in range(len(scores)):
                row = {
                    'impression_id': imp_id,
                    'candidate_idx': cand_idx
                }
                
                # Add predictions from each model
                for model_idx, pred_dict in enumerate(pred_dicts):
                    if imp_id in pred_dict and cand_idx < len(pred_dict[imp_id]):
                        row[f'pred_{model_idx}'] = pred_dict[imp_id][cand_idx]
                    else:
                        row[f'pred_{model_idx}'] = 0.0  # Default score
                
                rows.append(row)
    
    return pd.DataFrame(rows)


def compute_impression_auc(df: pd.DataFrame, pred_col: str = 'pred_ensemble') -> float:
    """
    Tính AUC trung bình theo impression (giống sugawarya)
    """
    aucs = []
    for imp_id, group in df.groupby('impression_id'):
        y_true = group['target'].values
        y_pred = group[pred_col].values
        
        # Skip nếu tất cả label giống nhau
        if len(np.unique(y_true)) < 2:
            continue
            
        try:
            auc = roc_auc_score(y_true, y_pred)
            aucs.append(auc)
        except:
            continue
    
    return np.mean(aucs) if aucs else 0.0


class WeightedMeanEnsemble:
    """
    Ensemble bằng weighted mean với tối ưu trọng số qua Optuna
    (Theo sugawarya/src/weighted_mean.py)
    """
    
    def __init__(self, n_trials: int = 200, sampling_rate: int = 100):
        self.n_trials = n_trials
        self.sampling_rate = sampling_rate  # Lấy mẫu mỗi N impression
        self.best_weights = None
        self.best_auc = None
        self.study = None
        self.training_history = []
        
    def fit(self, df: pd.DataFrame, pred_cols: List[str]):
        """
        Tối ưu trọng số trên validation set
        
        Args:
            df: DataFrame chứa predictions và targets
            pred_cols: List tên cột predictions (e.g., ['pred_0', 'pred_1', ...])
        """
        # Sampling để tăng tốc (giống sugawarya)
        unique_imps = df['impression_id'].unique()
        sampled_imps = unique_imps[::self.sampling_rate]
        mini_df = df[df['impression_id'].isin(sampled_imps)].copy()
        
        print(f"Optimizing on {len(sampled_imps)} impressions (sampled from {len(unique_imps)})")
        
        def objective(trial):
            weights = {}
            for pred_col in pred_cols:
                weights[pred_col] = trial.suggest_float(pred_col, 0.0, 1.0)
            
            # Weighted sum
            mini_df['pred_ensemble'] = sum(
                mini_df[col] * weight for col, weight in weights.items()
            )
            
            # Compute AUC
            score = compute_impression_auc(mini_df, 'pred_ensemble')
            return score
        
        # Optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.study = study
        self.best_weights = study.best_params
        self.best_auc = study.best_value
        
        # Save training history
        self.training_history = [
            {'trial': t.number, 'value': t.value, 'params': t.params}
            for t in study.trials
        ]
        
        print(f"\nBest AUC: {study.best_value:.4f}")
        print(f"Best weights: {self.best_weights}")
        
        return self
    
    def predict(self, df: pd.DataFrame, pred_cols: List[str]) -> np.ndarray:
        """
        Áp dụng weighted mean với trọng số đã tối ưu
        """
        if self.best_weights is None:
            raise ValueError("Must call fit() first")
        
        pred_ensemble = sum(
            df[col] * self.best_weights[col] for col in pred_cols
        )
        return pred_ensemble.values
    
    def save(self, save_dir: str):
        """
        Lưu weights và metadata
        
        Lưu:
        - weights.json: Best weights từ Optuna
        - metadata.json: Training config và metrics
        - study.pkl: Optuna study object (có thể visualize sau)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save weights
        weights_path = os.path.join(save_dir, 'weights.json')
        with open(weights_path, 'w') as f:
            json.dump(self.best_weights, f, indent=2)
        
        # Save metadata
        metadata = {
            'method': 'WeightedMean',
            'n_trials': self.n_trials,
            'sampling_rate': self.sampling_rate,
            'best_auc': self.best_auc,
            'best_weights': self.best_weights,
            'timestamp': datetime.now().isoformat(),
            'num_models': len(self.best_weights) if self.best_weights else 0
        }
        metadata_path = os.path.join(save_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save training history
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save Optuna study
        if self.study is not None:
            study_path = os.path.join(save_dir, 'optuna_study.pkl')
            with open(study_path, 'wb') as f:
                pickle.dump(self.study, f)
        
        print(f"\nWeightedMean saved to {save_dir}")
        print(f"  - weights.json")
        print(f"  - metadata.json")
        print(f"  - training_history.json")
        print(f"  - optuna_study.pkl")
    
    @classmethod
    def load(cls, save_dir: str):
        """
        Load weights từ file đã save
        """
        # Load metadata
        metadata_path = os.path.join(save_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls(
            n_trials=metadata['n_trials'],
            sampling_rate=metadata['sampling_rate']
        )
        instance.best_weights = metadata['best_weights']
        instance.best_auc = metadata['best_auc']
        
        # Load training history if exists
        history_path = os.path.join(save_dir, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                instance.training_history = json.load(f)
        
        # Load study if exists
        study_path = os.path.join(save_dir, 'optuna_study.pkl')
        if os.path.exists(study_path):
            with open(study_path, 'rb') as f:
                instance.study = pickle.load(f)
        
        print(f"WeightedMean loaded from {save_dir}")
        print(f"  Best AUC: {instance.best_auc:.4f}")
        print(f"  Weights: {instance.best_weights}")
        
        return instance


class StackingEnsemble:
    """
    LightGBM Stacking với feature engineering
    (Theo sugawarya/src/stacking.py)
    """
    
    def __init__(self, n_folds: int = 4, sampling_rate: int = 10):
        self.n_folds = n_folds
        self.sampling_rate = sampling_rate
        self.models = []
        self.feature_cols = None
        self.fold_metrics = []
        self.oof_auc = None
        self.lgb_params = None
        
    def _create_features(self, df: pd.DataFrame, pred_cols: List[str]) -> pd.DataFrame:
        """
        Feature engineering theo sugawarya:
        - Stats trong impression (mean, max, min, std)
        - Normalized scores (z-score, min-max)
        - Rank features
        - Pairwise features (diff, ratio, max, min)
        """
        df = df.copy()
        
        # 1. Basic stats trong impression
        for col in pred_cols:
            grouped = df.groupby('impression_id')[col]
            df[f'{col}_mean'] = grouped.transform('mean')
            df[f'{col}_max'] = grouped.transform('max')
            df[f'{col}_min'] = grouped.transform('min')
            df[f'{col}_std'] = grouped.transform('std').fillna(0)
        
        # 2. Normalized features
        for col in pred_cols:
            # Z-score normalization
            df[f'{col}_zscore'] = (
                (df[col] - df[f'{col}_mean']) / (df[f'{col}_std'] + 1e-8)
            )
            
            # Min-max normalization trong impression
            df[f'{col}_normed'] = (
                (df[col] - df[f'{col}_min']) / 
                (df[f'{col}_max'] - df[f'{col}_min'] + 1e-8)
            )
        
        # 3. Rank features
        for col in pred_cols:
            df[f'{col}_rank'] = df.groupby('impression_id')[col].rank()
            df[f'{col}_rank_desc'] = df.groupby('impression_id')[col].rank(ascending=False)
            
            # Normalized rank
            imp_count = df.groupby('impression_id')['impression_id'].transform('count')
            df[f'{col}_normedrank'] = df[f'{col}_rank'] / imp_count
            df[f'{col}_normedrank_desc'] = df[f'{col}_rank_desc'] / imp_count
        
        # 4. Pairwise features (chỉ lấy 1 vài cặp quan trọng để tránh quá nhiều features)
        if len(pred_cols) >= 2:
            col1, col2 = pred_cols[0], pred_cols[1]
            
            for suffix in ['', '_rank', '_zscore', '_normed']:
                c1 = col1 + suffix
                c2 = col2 + suffix
                
                if c1 in df.columns and c2 in df.columns:
                    df[f'feat_{c1}_{c2}_diff'] = df[c1] - df[c2]
                    df[f'feat_{c1}_{c2}_ratio'] = df[c1] / (df[c2] + 1e-8)
                    df[f'feat_{c1}_{c2}_max'] = np.maximum(df[c1], df[c2])
                    df[f'feat_{c1}_{c2}_min'] = np.minimum(df[c1], df[c2])
        
        # 5. Cross-model mean
        df['pred_mean_all'] = df[pred_cols].mean(axis=1)
        
        # 6. Impression count
        df['impression_count'] = df.groupby('impression_id')['impression_id'].transform('count')
        
        return df
    
    def fit(self, df: pd.DataFrame, pred_cols: List[str]):
        """
        Train LightGBM stacking models với GroupKFold
        """
        print("Creating features...")
        df = self._create_features(df, pred_cols)
        
        # Sampling để giảm thời gian train
        unique_imps = df['impression_id'].unique()
        sampled_imps = unique_imps[::self.sampling_rate]
        train_df = df[df['impression_id'].isin(sampled_imps)].copy()
        
        print(f"Training on {len(sampled_imps)} impressions (sampled from {len(unique_imps)})")
        
        # Define feature columns
        reserved_cols = ['impression_id', 'candidate_idx', 'target']
        self.feature_cols = [c for c in train_df.columns if c not in reserved_cols]
        
        # LightGBM params (LambdaRank cho ranking)
        lgb_params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_at': [5, 10],
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'max_bin': 1024,
            'verbose': -1,
            'seed': 42
        }
        self.lgb_params = lgb_params  # Save params
        
        # GroupKFold cross-validation
        gkf = GroupKFold(n_splits=self.n_folds)
        oof_preds = np.zeros(len(train_df))
        
        for fold, (train_idx, valid_idx) in enumerate(
            gkf.split(train_df, groups=train_df['impression_id'])
        ):
            print(f"\nFold {fold + 1}/{self.n_folds}")
            
            X_train = train_df.iloc[train_idx][self.feature_cols].values.astype(np.float32)
            y_train = train_df.iloc[train_idx]['target'].values.astype(np.float32)
            groups_train = train_df.iloc[train_idx].groupby('impression_id').size().values
            
            X_valid = train_df.iloc[valid_idx][self.feature_cols].values.astype(np.float32)
            y_valid = train_df.iloc[valid_idx]['target'].values.astype(np.float32)
            groups_valid = train_df.iloc[valid_idx].groupby('impression_id').size().values
            
            train_data = lgb.Dataset(X_train, y_train, group=groups_train)
            valid_data = lgb.Dataset(X_valid, y_valid, group=groups_valid)
            
            model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=10000,
                valid_sets=[valid_data],
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(50)
                ]
            )
            
            oof_preds[valid_idx] = model.predict(X_valid)
            self.models.append(model)
            
            # Save fold metrics
            fold_metric = {
                'fold': fold + 1,
                'best_iteration': model.best_iteration,
                'best_score': model.best_score
            }
            self.fold_metrics.append(fold_metric)
        
        # Compute OOF AUC
        train_df['pred_stacking'] = oof_preds
        oof_auc = compute_impression_auc(train_df, 'pred_stacking')
        self.oof_auc = oof_auc
        print(f"\nOOF AUC: {oof_auc:.4f}")
        
        return self
    
    def predict(self, df: pd.DataFrame, pred_cols: List[str]) -> np.ndarray:
        """
        Predict bằng ensemble của các fold models
        """
        if not self.models:
            raise ValueError("Must call fit() first")
        
        print("Creating features for inference...")
        df = self._create_features(df, pred_cols)
        
        X = df[self.feature_cols].values.astype(np.float32)
        
        # Average predictions from all folds
        preds = np.zeros(len(df))
        for model in tqdm(self.models, desc="Predicting"):
            preds += model.predict(X) / len(self.models)
        
        return preds
    
    def save(self, save_dir: str):
        """
        Lưu models và metadata
        
        Lưu:
        - fold_X.txt: LightGBM model cho mỗi fold
        - metadata.json: Training config và metrics
        - feature_cols.json: Danh sách features
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        for i, model in enumerate(self.models):
            model_path = os.path.join(save_dir, f'fold_{i+1}.txt')
            model.save_model(model_path)
        
        # Save metadata
        metadata = {
            'method': 'StackingLightGBM',
            'n_folds': self.n_folds,
            'sampling_rate': self.sampling_rate,
            'oof_auc': self.oof_auc,
            'lgb_params': self.lgb_params,
            'fold_metrics': self.fold_metrics,
            'num_features': len(self.feature_cols) if self.feature_cols else 0,
            'timestamp': datetime.now().isoformat()
        }
        metadata_path = os.path.join(save_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save feature columns
        if self.feature_cols:
            features_path = os.path.join(save_dir, 'feature_cols.json')
            with open(features_path, 'w') as f:
                json.dump(self.feature_cols, f, indent=2)
        
        print(f"\nStackingEnsemble saved to {save_dir}")
        print(f"  - {len(self.models)} model files (fold_X.txt)")
        print(f"  - metadata.json")
        print(f"  - feature_cols.json ({len(self.feature_cols)} features)")
    
    @classmethod
    def load(cls, save_dir: str):
        """
        Load models từ file đã save
        """
        # Load metadata
        metadata_path = os.path.join(save_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls(
            n_folds=metadata['n_folds'],
            sampling_rate=metadata['sampling_rate']
        )
        instance.oof_auc = metadata['oof_auc']
        instance.lgb_params = metadata['lgb_params']
        instance.fold_metrics = metadata['fold_metrics']
        
        # Load feature columns
        features_path = os.path.join(save_dir, 'feature_cols.json')
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                instance.feature_cols = json.load(f)
        
        # Load models
        instance.models = []
        for i in range(metadata['n_folds']):
            model_path = os.path.join(save_dir, f'fold_{i+1}.txt')
            if os.path.exists(model_path):
                model = lgb.Booster(model_file=model_path)
                instance.models.append(model)
        
        print(f"StackingEnsemble loaded from {save_dir}")
        print(f"  Loaded {len(instance.models)} fold models")
        print(f"  OOF AUC: {instance.oof_auc:.4f}")
        print(f"  Features: {len(instance.feature_cols)}")
        
        return instance


def save_prediction_file(df: pd.DataFrame, pred_col: str, output_path: str, 
                         format: str = 'rank', save_prod: bool = True):
    """
    Lưu predictions theo format submission
    
    Args:
        df: DataFrame with predictions
        pred_col: Column name for predictions
        output_path: Path to save file
        format: 'rank' for ranks, 'prod' for probabilities (default='rank')
        save_prod: If True, also save .prod file with probabilities (default=True)
    
    Format rank: <impression_id> [rank1,rank2,...]  (KHÔNG có dấu cách)
    Format prod: <impression_id> [prob1,prob2,...]  (xác suất thực)
    
    Ví dụ:
    - Scores: [0.8, 0.3, 0.9] (theo thứ tự candidate 0,1,2)
    - Rank output: [2,3,1]
    - Prod output: [0.8,0.3,0.9]
    """
    results_rank = []
    results_prod = []
    
    for imp_id, group in tqdm(df.groupby('impression_id'), desc=f"Generating {format}"):
        # Sắp xếp theo candidate_idx
        group = group.sort_values('candidate_idx')
        
        # Get probabilities
        probs = group[pred_col].tolist()
        
        # Format probabilities (prod format)
        probs_str = '[' + ','.join([f"{p:.6f}" for p in probs]) + ']'
        results_prod.append(f"{imp_id} {probs_str}\n")
        
        # Tính ranks từ probabilities
        ranks = group[pred_col].rank(ascending=False, method='first').astype(int).tolist()
        ranks_str = '[' + ','.join(map(str, ranks)) + ']'
        results_rank.append(f"{imp_id} {ranks_str}\n")
    
    # Save primary format
    if format == 'prod':
        with open(output_path, 'w') as f:
            f.writelines(results_prod)
        print(f"Saved probabilities to {output_path}")
        
        # Also save rank format
        if save_prod:
            rank_path = output_path.replace('.txt', '_rank.txt')
            with open(rank_path, 'w') as f:
                f.writelines(results_rank)
            print(f"Saved ranks to {rank_path}")
    else:  # format == 'rank'
        with open(output_path, 'w') as f:
            f.writelines(results_rank)
        print(f"Saved ranks to {output_path}")
        
        # Also save prod format
        if save_prod:
            prod_path = output_path.replace('.txt', '_prod.txt')
            if '_rank' in output_path:
                prod_path = output_path.replace('_rank.txt', '_prod.txt')
            with open(prod_path, 'w') as f:
                f.writelines(results_prod)
            print(f"Saved probabilities to {prod_path}")


if __name__ == '__main__':
    # Example usage
    print("Ensemble methods ready to use!")
    print("\nUsage:")
    print("1. Weighted Mean:")
    print("   ensemble = WeightedMeanEnsemble(n_trials=200)")
    print("   ensemble.fit(val_df, pred_cols)")
    print("   preds = ensemble.predict(test_df, pred_cols)")
    print("\n2. LightGBM Stacking:")
    print("   ensemble = StackingEnsemble(n_folds=4)")
    print("   ensemble.fit(val_df, pred_cols)")
    print("   preds = ensemble.predict(test_df, pred_cols)")
