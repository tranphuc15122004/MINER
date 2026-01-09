"""
Script để chạy ensemble trên MIND dataset

Workflow:
1. Load predictions từ nhiều models
2. Thử cả 2 phương pháp: Weighted Mean và Stacking
3. Đánh giá kết quả với sub_evaluator
"""
import argparse
import os
from ensemble import (
    load_predictions_as_df,
    WeightedMeanEnsemble,
    StackingEnsemble,
    save_prediction_file
)


def main():
    parser = argparse.ArgumentParser(description="Ensemble multiple model predictions")
    parser.add_argument('--predictions', nargs='+', required=True,
                        help='List of prediction files (e.g., pred1.txt pred2.txt pred3.txt)')
    parser.add_argument('--truth', required=True,
                        help='Truth file path')
    parser.add_argument('--method', choices=['weighted', 'stacking', 'both'], default='both',
                        help='Ensemble method to use')
    parser.add_argument('--output-dir', default='phase2/ensemble_results',
                        help='Output directory')
    parser.add_argument('--n-trials', type=int, default=500,
                        help='Number of Optuna trials for weighted mean')
    parser.add_argument('--n-folds', type=int, default=10,
                        help='Number of folds for stacking')
    parser.add_argument('--sampling-rate-weighted', type=int, default=100,
                        help='Sample every N impressions for weighted mean optimization')
    parser.add_argument('--sampling-rate-stacking', type=int, default=10,
                        help='Sample every N impressions for stacking training')
    
    args = parser.parse_args()
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("ENSEMBLE PREDICTIONS")
    print("="*80)
    print(f"Input predictions: {args.predictions}")
    print(f"Truth file: {args.truth}")
    print(f"Method: {args.method}")
    print(f"Output dir: {args.output_dir}")
    print("="*80)
    
    # Load data
    print("\n[1/4] Loading predictions and truth...")
    df = load_predictions_as_df(args.predictions, args.truth)
    pred_cols = [f'pred_{i}' for i in range(len(args.predictions))]
    
    print(f"Loaded {len(df)} samples from {df['impression_id'].nunique()} impressions")
    print(f"Prediction columns: {pred_cols}")
    
    # Split validation (80%) và test (20%) để tối ưu
    # Trong thực tế, bạn nên có validation set riêng
    unique_imps = df['impression_id'].unique()
    split_idx = int(len(unique_imps) * 0.8)
    val_imps = unique_imps[:split_idx]
    test_imps = unique_imps[split_idx:]
    
    val_df = df[df['impression_id'].isin(val_imps)].copy()
    test_df = df[df['impression_id'].isin(test_imps)].copy()
    
    print(f"Validation: {len(val_df)} samples, {len(val_imps)} impressions")
    print(f"Test: {len(test_df)} samples, {len(test_imps)} impressions")
    
    # Weighted Mean
    if args.method in ['weighted', 'both']:
        print("\n[2/4] Running Weighted Mean Ensemble...")
        print("-" * 80)
        
        wm_ensemble = WeightedMeanEnsemble(
            n_trials=args.n_trials,
            sampling_rate=args.sampling_rate_weighted
        )
        wm_ensemble.fit(val_df, pred_cols)
        
        # Save weights and metadata
        wm_save_dir = os.path.join(args.output_dir, 'weighted_mean')
        wm_ensemble.save(wm_save_dir)
        
        # Predict on full dataset
        full_df = df.copy()
        full_df['pred_weighted'] = wm_ensemble.predict(full_df, pred_cols)
        
        # Save prediction files (both rank and prod format)
        wm_output = os.path.join(args.output_dir, 'prediction_weighted_rank.txt')
        save_prediction_file(full_df, 'pred_weighted', wm_output, format='rank', save_prod=True)
    
    # Stacking
    if args.method in ['stacking', 'both']:
        print("\n[3/4] Running LightGBM Stacking Ensemble...")
        print("-" * 80)
        
        stacking_ensemble = StackingEnsemble(
            n_folds=args.n_folds,
            sampling_rate=args.sampling_rate_stacking
        )
        stacking_ensemble.fit(val_df, pred_cols)
        
        # Save models and metadata
        stacking_save_dir = os.path.join(args.output_dir, 'stacking')
        stacking_ensemble.save(stacking_save_dir)
        
        # Predict on full dataset
        full_df = df.copy()
        full_df['pred_stacking'] = stacking_ensemble.predict(full_df, pred_cols)
        
        # Save prediction files (both rank and prod format)
        stacking_output = os.path.join(args.output_dir, 'prediction_stacking_rank.txt')
        save_prediction_file(full_df, 'pred_stacking', stacking_output, format='rank', save_prod=True)
    
    print("\n[4/4] Done!")
    print("="*80)
    print("Ensemble predictions saved to:")
    if args.method in ['weighted', 'both']:
        print(f"  - Weighted Mean (rank): {wm_output}")
        print(f"  - Weighted Mean (prod): {wm_output.replace('_rank.txt', '_prod.txt')}")
        print(f"  - Models/weights: {wm_save_dir}")
    if args.method in ['stacking', 'both']:
        print(f"  - Stacking (rank): {stacking_output}")
        print(f"  - Stacking (prod): {stacking_output.replace('_rank.txt', '_prod.txt')}")
        print(f"  - Models/weights: {stacking_save_dir}")
    print("\nTo evaluate, run:")
    print(f"  python phase2/sub_evaluator.py --prediction-file <output_file> --truth-file {args.truth}")
    print("="*80)


if __name__ == '__main__':
    main()
