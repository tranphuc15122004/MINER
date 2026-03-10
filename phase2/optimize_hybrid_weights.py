"""
Grid Search để tối ưu trọng số kết hợp (alpha) cho Hybrid Ensemble

Script này:
1. Load WeightedMean và Stacking models đã train
2. Thử các giá trị alpha khác nhau (0.0 -> 1.0)
3. Tính AUC cho mỗi alpha trên validation set
4. Lưu alpha tối ưu và visualization

Usage:
    python phase2/optimize_hybrid_weights.py \
        --predictions pred1_prod.txt pred2_prod.txt pred3_prod.txt \
        --truth truth.txt \
        --weighted-dir phase2/ensemble_results/weighted_mean \
        --stacking-dir phase2/ensemble_results/stacking \
        --output-dir phase2/ensemble_results/hybrid \
        --alpha-min 0.0 \
        --alpha-max 1.0 \
        --alpha-step 0.05
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.ensemble import (
    WeightedMeanEnsemble,
    StackingEnsemble,
    load_predictions_as_df,
    compute_impression_auc
)


def grid_search_alpha(df, pred_cols, wm_model, stacking_model, alpha_range, sampling_rate=1):
    """
    Grid search để tìm alpha tối ưu
    
    Args:
        df: DataFrame với predictions và targets
        pred_cols: List tên cột predictions
        wm_model: WeightedMeanEnsemble đã train
        stacking_model: StackingEnsemble đã train
        alpha_range: List các giá trị alpha cần thử
        sampling_rate: Lấy mẫu 1/N impressions (default=1, tức là dùng tất cả)
        
    Returns:
        dict với results cho mỗi alpha
    """
    print("\n" + "="*80)
    print("GRID SEARCH FOR OPTIMAL ALPHA")
    print("="*80)
    print(f"Alpha range: {alpha_range[0]:.3f} -> {alpha_range[-1]:.3f} (step={alpha_range[1]-alpha_range[0]:.3f})")
    print(f"Total trials: {len(alpha_range)}")
    
    # Apply sampling if needed
    total_impressions = df['impression_id'].nunique()
    if sampling_rate > 1:
        unique_imps = df['impression_id'].unique()
        sampled_imps = unique_imps[::sampling_rate]
        df = df[df['impression_id'].isin(sampled_imps)].copy()
        print(f"Sampling: Using {len(sampled_imps)}/{total_impressions} impressions (every {sampling_rate}th)")
        print(f"Validation set: {len(df)} rows, {len(sampled_imps)} impressions (sampled)")
    else:
        print(f"Validation set: {len(df)} rows, {total_impressions} impressions (full)")
    
    print("="*80)
    
    # Get predictions từ 2 models
    print("\nGenerating predictions from base models...")
    print("  [1/2] WeightedMean predicting...")
    wm_preds = wm_model.predict(df, pred_cols)
    print(f"        ✓ Done (trained AUC: {wm_model.best_auc:.4f})")
    
    print("  [2/2] Stacking predicting...")
    stacking_preds = stacking_model.predict(df, pred_cols)
    print(f"        ✓ Done (OOF AUC: {stacking_model.oof_auc:.4f})")
    
    # Grid search
    print(f"\nTesting {len(alpha_range)} alpha values...")
    results = []
    best_auc = -1
    best_alpha = None
    
    for i, alpha in enumerate(alpha_range):
        # Combine predictions
        hybrid_preds = alpha * wm_preds + (1 - alpha) * stacking_preds
        
        # Compute AUC
        df_temp = df.copy()
        df_temp['pred_hybrid'] = hybrid_preds
        auc = compute_impression_auc(df_temp, 'pred_hybrid')
        
        results.append({
            'alpha': alpha,
            'auc': auc,
            'weighted_weight': alpha,
            'stacking_weight': 1 - alpha
        })
        
        # Track best
        if auc > best_auc:
            best_auc = auc
            best_alpha = alpha
        
        # Progress
        if (i + 1) % 5 == 0 or i == 0 or i == len(alpha_range) - 1:
            print(f"  [{i+1}/{len(alpha_range)}] alpha={alpha:.3f} -> AUC={auc:.4f} {'🏆' if alpha == best_alpha else ''}")
    
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETED!")
    print("="*80)
    print(f"Best alpha: {best_alpha:.3f}")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"  - WeightedMean weight: {best_alpha:.1%}")
    print(f"  - Stacking weight: {1-best_alpha:.1%}")
    print("\nComparison:")
    print(f"  vs WeightedMean alone (α=1.0): {best_auc - wm_model.best_auc:+.4f}")
    print(f"  vs Stacking alone (α=0.0): {best_auc - stacking_model.oof_auc:+.4f}")
    
    # Tìm improvement so với 50-50
    default_result = [r for r in results if abs(r['alpha'] - 0.5) < 0.01]
    if default_result:
        default_auc = default_result[0]['auc']
        print(f"  vs Default (α=0.5): {best_auc - default_auc:+.4f}")
    
    print("="*80)
    
    return {
        'results': results,
        'best_alpha': best_alpha,
        'best_auc': best_auc,
        'wm_baseline_auc': wm_model.best_auc,
        'stacking_baseline_auc': stacking_model.oof_auc
    }


def plot_alpha_curve(results, output_dir):
    """
    Vẽ biểu đồ AUC vs Alpha
    """
    alphas = [r['alpha'] for r in results['results']]
    aucs = [r['auc'] for r in results['results']]
    
    plt.figure(figsize=(12, 6))
    
    # Main curve
    plt.plot(alphas, aucs, 'b-', linewidth=2, label='Hybrid AUC')
    
    # Best point
    best_idx = alphas.index(results['best_alpha'])
    plt.scatter([results['best_alpha']], [aucs[best_idx]], 
                color='red', s=200, zorder=5, marker='*',
                label=f"Best α={results['best_alpha']:.3f}, AUC={results['best_auc']:.4f}")
    
    # Baselines
    plt.axhline(y=results['wm_baseline_auc'], color='green', linestyle='--', 
                alpha=0.7, label=f'WeightedMean baseline (AUC={results["wm_baseline_auc"]:.4f})')
    plt.axhline(y=results['stacking_baseline_auc'], color='orange', linestyle='--',
                alpha=0.7, label=f'Stacking baseline (AUC={results["stacking_baseline_auc"]:.4f})')
    
    # Default α=0.5
    default_result = [r for r in results['results'] if abs(r['alpha'] - 0.5) < 0.01]
    if default_result:
        plt.scatter([0.5], [default_result[0]['auc']], 
                   color='purple', s=100, zorder=4, marker='o',
                   label=f"Default α=0.5, AUC={default_result[0]['auc']:.4f}")
    
    plt.xlabel('Alpha (Weight for WeightedMean)', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.title('Grid Search: Hybrid Ensemble Alpha Optimization', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    plot_path = os.path.join(output_dir, 'alpha_optimization_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to {plot_path}")
    plt.close()


def save_results(results, output_dir, args):
    """
    Lưu kết quả grid search
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Best alpha config
    best_config = {
        'best_alpha': results['best_alpha'],
        'best_auc': results['best_auc'],
        'weighted_weight': results['best_alpha'],
        'stacking_weight': 1 - results['best_alpha'],
        'wm_baseline_auc': results['wm_baseline_auc'],
        'stacking_baseline_auc': results['stacking_baseline_auc'],
        'improvement_vs_wm': results['best_auc'] - results['wm_baseline_auc'],
        'improvement_vs_stacking': results['best_auc'] - results['stacking_baseline_auc'],
        'sampling_rate': args.sampling_rate if hasattr(args, 'sampling_rate') else 1,
        'timestamp': datetime.now().isoformat(),
        'args': vars(args)
    }
    
    config_path = os.path.join(output_dir, 'best_alpha.json')
    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"✓ Best alpha config saved to {config_path}")
    
    # 2. Full results table
    results_df = pd.DataFrame(results['results'])
    results_df = results_df.sort_values('auc', ascending=False)
    
    csv_path = os.path.join(output_dir, 'grid_search_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Full results saved to {csv_path}")
    
    # 3. Top 10 alphas
    top10 = results_df.head(10)
    print("\n" + "="*80)
    print("TOP 10 ALPHA VALUES")
    print("="*80)
    print(top10.to_string(index=False))
    print("="*80)
    
    # 4. Summary metadata
    metadata = {
        'method': 'HybridEnsemble',
        'optimization': 'GridSearch',
        'best_alpha': results['best_alpha'],
        'best_auc': results['best_auc'],
        'num_trials': len(results['results']),
        'alpha_range': {
            'min': min(r['alpha'] for r in results['results']),
            'max': max(r['alpha'] for r in results['results']),
            'step': results['results'][1]['alpha'] - results['results'][0]['alpha'] if len(results['results']) > 1 else 0
        },
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Grid Search for Hybrid Ensemble Alpha Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full data (slow but accurate)
  python phase2/optimize_hybrid_weights.py \\
      --predictions checkpoint/prediction_prod_Ngoc.txt \\
                    checkpoint/prediction_prod_Phuc.txt \\
                    checkpoint/prediction_prod_Son.txt \\
      --truth phase2/ref/truth.txt \\
      --weighted-dir phase2/ensemble_results/weighted_mean \\
      --stacking-dir phase2/ensemble_results/stacking \\
      --output-dir phase2/ensemble_results/hybrid \\
      --alpha-min 0.0 \\
      --alpha-max 1.0 \\
      --alpha-step 0.05
  
  # Fast mode: sample 10% of data (10x faster)
  python phase2/optimize_hybrid_weights.py \\
      --predictions checkpoint/prediction_prod_Ngoc.txt \\
                    checkpoint/prediction_prod_Phuc.txt \\
                    checkpoint/prediction_prod_Son.txt \\
      --truth phase2/ref/truth.txt \\
      --weighted-dir phase2/ensemble_results/weighted_mean \\
      --stacking-dir phase2/ensemble_results/stacking \\
      --output-dir phase2/ensemble_results/hybrid \\
      --sampling-rate 10
        """
    )
    
    
    # Input
    parser.add_argument('--predictions', nargs='+', required=True,
                        help='Prediction files (prod format)')
    parser.add_argument('--truth', required=True,
                        help='Truth file for validation')
    
    # Model directories
    parser.add_argument('--weighted-dir', required=True,
                        help='WeightedMean model directory')
    parser.add_argument('--stacking-dir', required=True,
                        help='Stacking model directory')
    
    # Alpha range
    parser.add_argument('--alpha-min', type=float, default=0.0,
                        help='Minimum alpha value (default=0.0)')
    parser.add_argument('--alpha-max', type=float, default=1.0,
                        help='Maximum alpha value (default=1.0)')
    parser.add_argument('--alpha-step', type=float, default=0.05,
                        help='Step size for alpha grid (default=0.05)')
    
    # Sampling for speed
    parser.add_argument('--sampling-rate', type=int, default=1,
                        help='Sample every Nth impression for faster evaluation (default=1, use all). E.g., 10 = use 10%% of data')
    
    # Output
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*80)
    print("HYBRID ENSEMBLE ALPHA OPTIMIZATION")
    print("="*80)
    print(f"Input predictions: {len(args.predictions)} files")
    for i, pred in enumerate(args.predictions):
        print(f"  [{i}] {pred}")
    if args.sampling_rate > 1:
        print(f"⚡ Fast mode: Sampling rate 1/{args.sampling_rate} (using ~{100/args.sampling_rate:.1f}% of data)")
    print(f"Truth file: {args.truth}")
    print(f"WeightedMean model: {args.weighted_dir}")
    print(f"Stacking model: {args.stacking_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Load predictions
    print("\n[STEP 1] Loading predictions and truth...")
    df = load_predictions_as_df(args.predictions, args.truth, auto_convert_rank=True)
    pred_cols = [f'pred_{i}' for i in range(len(args.predictions))]
    
    print(f"✓ Loaded {len(df)} rows from {df['impression_id'].nunique()} impressions")
    print(f"✓ Prediction columns: {pred_cols}")
    
    if 'target' not in df.columns:
        raise ValueError("Truth file required for optimization! Use --truth flag.")
    
    pos_rate = df['target'].mean()
    print(f"✓ Positive rate: {pos_rate:.2%} ({df['target'].sum()}/{len(df)})")
    
    # Load models
    print("\n[STEP 2] Loading trained models...")
    print(f"  Loading WeightedMean from {args.weighted_dir}...")
    wm_model = WeightedMeanEnsemble.load(args.weighted_dir)
    print(f"  ✓ WeightedMean loaded (AUC: {wm_model.best_auc:.4f})")
    
    print(f"  Loading Stacking from {args.stacking_dir}...")
    stacking_model = StackingEnsemble.load(args.stacking_dir)
    print(f"  ✓ Stacking loaded (OOF AUC: {stacking_model.oof_auc:.4f})")
    
    # Create alpha range
    alpha_range = np.arange(args.alpha_min, args.alpha_max + args.alpha_step/2, args.alpha_step)
    alpha_range = np.round(alpha_range, 3)  # Avoid floating point errors
    
    # Grid search
    print("\n[STEP 3] Grid Search...")
    results = grid_search_alpha(df, pred_cols, wm_model, stacking_model, alpha_range,
                                sampling_rate=args.sampling_rate)
    
    # Save results
    print("\n[STEP 4] Saving results...")
    save_results(results, args.output_dir, args)
    
    # Plot
    print("\n[STEP 5] Generating visualization...")
    plot_alpha_curve(results, args.output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("✅ OPTIMIZATION COMPLETED!")
    print("="*80)
    print(f"Best alpha: {results['best_alpha']:.3f}")
    print(f"  WeightedMean: {results['best_alpha']:.1%}")
    print(f"  Stacking: {1-results['best_alpha']:.1%}")
    print(f"Best AUC: {results['best_auc']:.4f}")
    print(f"\nOutput directory: {args.output_dir}")
    print("  - best_alpha.json (use this for inference)")
    print("  - grid_search_results.csv (all trials)")
    print("  - alpha_optimization_curve.png (visualization)")
    print("  - metadata.json")
    print("\nTo use optimal alpha in inference:")
    print(f"  python phase2/universal_infer.py \\")
    print(f"      --predictions <your_test_predictions> \\")
    print(f"      --weighted-dir {args.weighted_dir} \\")
    print(f"      --stacking-dir {args.stacking_dir} \\")
    print(f"      --alpha-file {args.output_dir}/best_alpha.json \\")
    print(f"      --output-dir <output_dir> \\")
    print(f"      --methods hybrid")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
