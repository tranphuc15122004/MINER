"""
Universal Inference Script - Ensemble 3 Methods (Weighted, Stacking, Hybrid)

Chức năng:
- Nhận file dự đoán dạng XÁC SUẤT (prod format)
- Tự động load và chạy 3 ensemble methods:
  1. WeightedMean (Optuna)
  2. Stacking (LightGBM)
  3. Hybrid (Average của 2 methods trên)
- Output: Rank predictions cho cả 3 methods

Usage:
    python phase2/universal_infer.py \
        --predictions pred1_prod.txt pred2_prod.txt pred3_prod.txt \
        --weighted-dir phase2/ensemble_results/weighted_mean \
        --stacking-dir phase2/ensemble_results/stacking \
        --output-dir results \
        --truth truth.txt  # Optional

    # Hoặc chỉ 1 method:
    python phase2/universal_infer.py \
        --predictions pred1_prod.txt pred2_prod.txt \
        --weighted-dir phase2/ensemble_results/weighted_mean \
        --output-dir results \
        --methods weighted

    # Hoặc cả 3:
    python phase2/universal_infer.py \
        --predictions pred1_prod.txt pred2_prod.txt \
        --weighted-dir phase2/ensemble_results/weighted_mean \
        --stacking-dir phase2/ensemble_results/stacking \
        --output-dir results \
        --methods all  # weighted,stacking,hybrid
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.ensemble import (
    WeightedMeanEnsemble,
    StackingEnsemble,
    load_predictions_as_df,
    save_prediction_file,
    compute_impression_auc
)


def run_weighted_mean(df, pred_cols, model_dir, output_path):
    """Run WeightedMean ensemble"""
    print("\n" + "="*80)
    print("[METHOD 1/3] WEIGHTED MEAN (Optuna-optimized)")
    print("="*80)
    
    # Load model
    print(f"Loading model from {model_dir}...")
    model = WeightedMeanEnsemble.load(model_dir)
    print(f"✓ Loaded (AUC: {model.best_auc:.4f})")
    
    # Predict
    print("Predicting...")
    preds = model.predict(df, pred_cols)
    df_out = df.copy()
    df_out['pred_weighted'] = preds
    
    # Compute AUC if truth available
    if 'target' in df.columns:
        auc = compute_impression_auc(df_out, 'pred_weighted')
        print(f"✓ AUC on test set: {auc:.4f}")
    
    # Save
    save_prediction_file(df_out, 'pred_weighted', output_path, format='rank', save_prod=True)
    print(f"✓ Saved to {output_path}")
    
    return preds


def run_stacking(df, pred_cols, model_dir, output_path):
    """Run Stacking ensemble"""
    print("\n" + "="*80)
    print("[METHOD 2/3] STACKING (LightGBM)")
    print("="*80)
    
    # Load model
    print(f"Loading model from {model_dir}...")
    model = StackingEnsemble.load(model_dir)
    print(f"✓ Loaded (OOF AUC: {model.oof_auc:.4f})")
    
    # Predict
    print("Predicting...")
    preds = model.predict(df, pred_cols)
    df_out = df.copy()
    df_out['pred_stacking'] = preds
    
    # Compute AUC if truth available
    if 'target' in df.columns:
        auc = compute_impression_auc(df_out, 'pred_stacking')
        print(f"✓ AUC on test set: {auc:.4f}")
    
    # Save
    save_prediction_file(df_out, 'pred_stacking', output_path, format='rank', save_prod=True)
    print(f"✓ Saved to {output_path}")
    
    return preds


def run_hybrid(df, pred_cols, weighted_dir, stacking_dir, output_path, alpha=0.5):
    """Run Hybrid ensemble (average của WeightedMean + Stacking)"""
    print("\n" + "="*80)
    print(f"[METHOD 3/3] HYBRID (WeightedMean + Stacking, alpha={alpha:.2f})")
    print("="*80)
    
    # Load models
    print(f"Loading WeightedMean from {weighted_dir}...")
    wm_model = WeightedMeanEnsemble.load(weighted_dir)
    print(f"✓ WeightedMean loaded (AUC: {wm_model.best_auc:.4f})")
    
    print(f"Loading Stacking from {stacking_dir}...")
    stacking_model = StackingEnsemble.load(stacking_dir)
    print(f"✓ Stacking loaded (OOF AUC: {stacking_model.oof_auc:.4f})")
    
    # Predict
    print(f"Predicting with WeightedMean...")
    wm_preds = wm_model.predict(df, pred_cols)
    
    print(f"Predicting with Stacking...")
    stacking_preds = stacking_model.predict(df, pred_cols)
    
    # Hybrid: weighted average
    print(f"Combining predictions (alpha={alpha:.2f})...")
    hybrid_preds = alpha * wm_preds + (1 - alpha) * stacking_preds
    
    df_out = df.copy()
    df_out['pred_hybrid'] = hybrid_preds
    
    # Compute AUC if truth available
    if 'target' in df.columns:
        auc = compute_impression_auc(df_out, 'pred_hybrid')
        print(f"✓ Hybrid AUC: {auc:.4f}")
        print(f"  vs WeightedMean: {auc - wm_model.best_auc:+.4f}")
        print(f"  vs Stacking: {auc - stacking_model.oof_auc:+.4f}")
    
    # Save
    save_prediction_file(df_out, 'pred_hybrid', output_path, format='rank', save_prod=True)
    print(f"✓ Saved to {output_path}")
    
    return hybrid_preds


def main():
    parser = argparse.ArgumentParser(
        description="Universal Inference - 3 Ensemble Methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All 3 methods
  python phase2/universal_infer.py \\
      --predictions pred1.txt pred2.txt pred3.txt \\
      --weighted-dir phase2/ensemble_results/weighted_mean \\
      --stacking-dir phase2/ensemble_results/stacking \\
      --output-dir results \\
      --methods all

  # Only WeightedMean
  python phase2/universal_infer.py \\
      --predictions pred1.txt pred2.txt \\
      --weighted-dir phase2/ensemble_results/weighted_mean \\
      --output-dir results \\
      --methods weighted

  # WeightedMean + Stacking (no Hybrid)
  python phase2/universal_infer.py \\
      --predictions pred1.txt pred2.txt \\
      --weighted-dir phase2/ensemble_results/weighted_mean \\
      --stacking-dir phase2/ensemble_results/stacking \\
      --output-dir results \\
      --methods weighted,stacking
        """
    )
    
    # Input
    parser.add_argument('--predictions', nargs='+', required=True,
                        help='Prediction files (prod format - xác suất)')
    parser.add_argument('--truth', default=None,
                        help='Truth file (optional, for AUC calculation)')
    
    # Model directories
    parser.add_argument('--weighted-dir', default=None,
                        help='WeightedMean model directory')
    parser.add_argument('--stacking-dir', default=None,
                        help='Stacking model directory')
    
    # Methods
    parser.add_argument('--methods', default='all',
                        help='Methods to run: all, weighted, stacking, hybrid, or comma-separated (e.g., weighted,stacking)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Alpha for Hybrid method (weight for WeightedMean). Default=0.5')
    
    # Output
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for predictions')
    
    args = parser.parse_args()
    
    # Parse methods
    if args.methods == 'all':
        methods = ['weighted', 'stacking', 'hybrid']
    else:
        methods = [m.strip() for m in args.methods.split(',')]
    
    # Validate
    if 'weighted' in methods and not args.weighted_dir:
        parser.error("--weighted-dir required for WeightedMean method")
    if 'stacking' in methods and not args.stacking_dir:
        parser.error("--stacking-dir required for Stacking method")
    if 'hybrid' in methods:
        if not args.weighted_dir or not args.stacking_dir:
            parser.error("--weighted-dir and --stacking-dir required for Hybrid method")
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print header
    print("\n" + "="*80)
    print("UNIVERSAL INFERENCE - 3 ENSEMBLE METHODS")
    print("="*80)
    print(f"Input predictions: {len(args.predictions)} files")
    for i, pred in enumerate(args.predictions):
        print(f"  [{i}] {pred}")
    
    # Mode indicator
    if args.truth:
        print(f"Mode: EVALUATION (with ground truth)")
        print(f"Truth file: {args.truth}")
    else:
        print(f"Mode: INFERENCE ONLY (no ground truth - production mode)")
        print(f"Truth file: Not provided")
    
    print(f"Methods to run: {', '.join(methods)}")
    print(f"Output directory: {args.output_dir}")
    if 'hybrid' in methods:
        print(f"Hybrid alpha: {args.alpha:.2f} (WeightedMean={args.alpha:.0%}, Stacking={1-args.alpha:.0%})")
    print("="*80)
    
    # Load predictions
    print("\n[STEP 1] Loading prediction files...")
    df = load_predictions_as_df(args.predictions, args.truth)
    pred_cols = [f'pred_{i}' for i in range(len(args.predictions))]
    
    print(f"✓ Loaded {len(df)} rows from {df['impression_id'].nunique()} impressions")
    print(f"✓ Prediction columns: {pred_cols}")
    
    if 'target' in df.columns:
        pos_rate = df['target'].mean()
        print(f"✓ Ground truth available: {pos_rate:.2%} positive ({df['target'].sum()}/{len(df)})")
        print(f"✓ AUC will be computed for each method")
    else:
        print(f"✓ Inference-only mode (no ground truth)")
        print(f"✓ Only predictions will be generated")
    
    # Run methods
    results = {}
    
    if 'weighted' in methods:
        output_path = os.path.join(args.output_dir, 'prediction_weighted_rank.txt')
        results['weighted'] = run_weighted_mean(df, pred_cols, args.weighted_dir, output_path)
    
    if 'stacking' in methods:
        output_path = os.path.join(args.output_dir, 'prediction_stacking_rank.txt')
        results['stacking'] = run_stacking(df, pred_cols, args.stacking_dir, output_path)
    
    if 'hybrid' in methods:
        output_path = os.path.join(args.output_dir, 'prediction_hybrid_rank.txt')
        results['hybrid'] = run_hybrid(
            df, pred_cols, args.weighted_dir, args.stacking_dir, 
            output_path, alpha=args.alpha
        )
    
    # Summary
    print("\n" + "="*80)
    print("✅ UNIVERSAL INFERENCE COMPLETED!")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print("\nGenerated files:")
    
    if 'weighted' in methods:
        print(f"  [WeightedMean]")
        print(f"    - Rank: {args.output_dir}/prediction_weighted_rank.txt")
        print(f"    - Prod: {args.output_dir}/prediction_weighted_prod.txt")
    
    if 'stacking' in methods:
        print(f"  [Stacking]")
        print(f"    - Rank: {args.output_dir}/prediction_stacking_rank.txt")
        print(f"    - Prod: {args.output_dir}/prediction_stacking_prod.txt")
    
    if 'hybrid' in methods:
        print(f"  [Hybrid (alpha={args.alpha:.2f})]")
        print(f"    - Rank: {args.output_dir}/prediction_hybrid_rank.txt")
        print(f"    - Prod: {args.output_dir}/prediction_hybrid_prod.txt")
    
    if args.truth:
        print("\n📊 Ground truth was provided - AUC scores computed above")
        print("\nTo run formal evaluation with sub_evaluator:")
        print(f"  python phase2/sub_evaluator.py . {args.output_dir}_eval \\")
        print(f"      --prediction-file {args.output_dir}/prediction_<method>_rank.txt \\")
        print(f"      --truth-file {args.truth}")
    else:
        print("\n💡 Inference-only mode - No ground truth provided")
        print("   Predictions generated successfully!")
        print("   To evaluate later with ground truth, use sub_evaluator.py")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
