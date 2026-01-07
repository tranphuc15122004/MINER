"""
Script để load saved ensemble models và inference trên test set mới

✨ Universal script - Tự động detect loại model (Weighted Mean hoặc Stacking)

Usage Examples:

  # Weighted Mean model
  python phase2/load_and_predict.py \
      --model-dir phase2/ensemble_results/weighted_mean \
      --predictions pred1.txt pred2.txt pred3.txt \
      --output weighted_output.txt
  
  # Stacking model - CÙNG SCRIPT, tự động detect!
  python phase2/load_and_predict.py \
      --model-dir phase2/ensemble_results/stacking \
      --predictions pred1.txt pred2.txt pred3.txt \
      --output stacking_output.txt
  
  # Với truth file (để evaluate sau)
  python phase2/load_and_predict.py \
      --model-dir phase2/ensemble_results/weighted_mean \
      --predictions pred1.txt pred2.txt \
      --truth truth.txt \
      --output output.txt

Note: Script tự động phát hiện loại model từ metadata.json
      Không cần chỉ định --method weighted hay --method stacking
"""
import argparse
import os
from ensemble import (
    WeightedMeanEnsemble,
    StackingEnsemble,
    load_predictions_as_df,
    save_prediction_file
)


def main():
    parser = argparse.ArgumentParser(description="Load saved ensemble and predict")
    parser.add_argument('--model-dir', required=True,
                        help='Directory containing saved model')
    parser.add_argument('--predictions', nargs='+', required=True,
                        help='Prediction files for test set')
    parser.add_argument('--truth', default=None,
                        help='Truth file for test set (optional - only needed for evaluation)')
    parser.add_argument('--output', default='prediction_output_rank.txt',
                        help='Output prediction file')
    parser.add_argument('--method', choices=['weighted', 'stacking'],
                        help='Model type (auto-detect from metadata if not specified)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("LOAD ENSEMBLE MODEL AND PREDICT")
    print("="*80)
    print(f"Model dir: {args.model_dir}")
    print(f"Predictions: {args.predictions}")
    print(f"Truth: {args.truth if args.truth else 'Not provided (inference only)'}")
    print(f"Output: {args.output}")
    print("="*80)
    
    # Auto-detect model type from metadata
    import json
    metadata_path = os.path.join(args.model_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    model_type = metadata['method']
    print(f"\n🔍 Auto-detected model type: {model_type}")
    print(f"   (No need to specify --method, script auto-detects from metadata.json)")
    
    # Load model
    print(f"\n📦 Loading model from {args.model_dir}...")
    if model_type == 'WeightedMean':
        model = WeightedMeanEnsemble.load(args.model_dir)
        print(f"   ✓ Loaded Weighted Mean Ensemble (Optuna-optimized)")
    elif model_type == 'StackingLightGBM':
        model = StackingEnsemble.load(args.model_dir)
        print(f"   ✓ Loaded LightGBM Stacking Ensemble ({metadata['n_folds']}-fold)")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load test data
    print("\nLoading test data...")
    df = load_predictions_as_df(args.predictions, args.truth)
    pred_cols = [f'pred_{i}' for i in range(len(args.predictions))]
    
    if args.truth:
        print(f"Test data: {len(df)} samples, {df['impression_id'].nunique()} impressions (with ground truth)")
    else:
        print(f"Test data: {len(df)} samples, {df['impression_id'].nunique()} impressions (inference only)")
    
    # Predict
    print("\nPredicting...")
    predictions = model.predict(df, pred_cols)
    df['pred_ensemble'] = predictions
    
    # Save
    print(f"\nSaving predictions to {args.output}...")
    save_prediction_file(df, 'pred_ensemble', args.output)
    
    print("\n" + "="*80)
    print("Done!")
    if args.truth:
        print(f"\nTo evaluate, run:")
        print(f"  python phase2/sub_evaluator.py --prediction-file {args.output} --truth-file {args.truth}")
    else:
        print(f"\nPrediction file generated: {args.output}")
        print("To evaluate later, run sub_evaluator with a truth file.")
    print("="*80)


if __name__ == '__main__':
    main()
