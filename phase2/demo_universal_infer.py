"""
Demo: Universal Inference với 3 Ensemble Methods

Test script để demonstrate cách sử dụng universal_infer.py

Modes:
- EVALUATION mode: Với truth file (để tính AUC)
- INFERENCE mode: Không có truth file (production mode)

Usage:
    python phase2/demo_universal_infer.py [--mode eval|infer]
"""
import os
import sys
import argparse

def print_section(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Demo Universal Inference')
    parser.add_argument('--mode', choices=['eval', 'infer'], default='infer',
                        help='eval: với truth file, infer: không có truth (default: infer)')
    args = parser.parse_args()
    
    print_section("DEMO: UNIVERSAL INFERENCE - 3 ENSEMBLE METHODS")
    
    # Paths
    predictions = [
        'checkpoint/prediction_P.txt',
        'checkpoint/prediction_S.txt',
        'checkpoint/prediction_N.txt'  
    ]
    
    weighted_dir = 'phase2/ensemble_results/weighted_mean'
    stacking_dir = 'phase2/ensemble_results/stacking'
    truth_file = 'phase2/ref/truth.txt' if args.mode == 'eval' else None
    output_dir = f'phase2/ensemble_results/final_{args.mode}'
    
    print(f"\n📋 Configuration (Mode: {args.mode.upper()}):")
    print(f"  Predictions: {len(predictions)} files")
    for i, p in enumerate(predictions):
        print(f"    [{i}] {p}")
    print(f"  WeightedMean dir: {weighted_dir}")
    print(f"  Stacking dir: {stacking_dir}")
    
    if args.mode == 'eval':
        print(f"  Truth file: {truth_file} (EVALUATION mode - will compute AUC)")
    else:
        print(f"  Truth file: None (INFERENCE mode - production scenario)")
    
    print(f"  Output dir: {output_dir}")
    
    # Check if models exist
    print("\n🔍 Checking models...")
    if not os.path.exists(weighted_dir):
        print(f"❌ WeightedMean model not found: {weighted_dir}")
        print("   Run 'python phase2/run_ensemble.py' first to train models!")
        return
    
    if not os.path.exists(stacking_dir):
        print(f"❌ Stacking model not found: {stacking_dir}")
        print("   Run 'python phase2/run_ensemble.py' first to train models!")
        return
    
    print("✓ WeightedMean model exists")
    print("✓ Stacking model exists")
    
    # Build command (single line for Windows compatibility)
    cmd_parts = [
        'python phase2/universal_infer.py',
        '--predictions', ' '.join(predictions),
        '--weighted-dir', weighted_dir,
        '--stacking-dir', stacking_dir,
    ]
    
    # Add truth only in eval mode
    if args.mode == 'eval':
        cmd_parts.extend(['--truth', truth_file])
    
    cmd_parts.extend([
        '--output-dir', output_dir,
        '--methods', 'all',
        '--alpha', '0.5'
    ])
    
    cmd = ' '.join(cmd_parts)
    
    print_section("COMMAND TO RUN")
    # Print formatted for readability
    cmd_display = f"""python phase2/universal_infer.py \\
    --predictions {' '.join(predictions)} \\
    --weighted-dir {weighted_dir} \\
    --stacking-dir {stacking_dir} \\"""
    
    if args.mode == 'eval':
        cmd_display += f"""
    --truth {truth_file} \\"""
    
    cmd_display += f"""
    --output-dir {output_dir} \\
    --methods all \\
    --alpha 0.5"""
    
    print(cmd_display)
    
    print_section("RUNNING INFERENCE...")
    
    # Run command
    import subprocess
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode == 0:
        print_section("✅ SUCCESS!")
        
        # Show output files
        print("\n📁 Generated files:")
        if os.path.exists(output_dir):
            for f in sorted(os.listdir(output_dir)):
                fpath = os.path.join(output_dir, f)
                size = os.path.getsize(fpath)
                print(f"  - {f} ({size:,} bytes)")
        
        if args.mode == 'eval':
            print_section("NEXT STEPS - EVALUATION MODE")
            print("\n1. Evaluate WeightedMean:")
            print(f"   python phase2/sub_evaluator.py . eval_weighted \\")
            print(f"       --prediction-file {output_dir}/prediction_weighted_rank.txt \\")
            print(f"       --truth-file {truth_file}")
            
            print("\n2. Evaluate Stacking:")
            print(f"   python phase2/sub_evaluator.py . eval_stacking \\")
            print(f"       --prediction-file {output_dir}/prediction_stacking_rank.txt \\")
            print(f"       --truth-file {truth_file}")
            
            print("\n3. Evaluate Hybrid:")
            print(f"   python phase2/sub_evaluator.py . eval_hybrid \\")
            print(f"       --prediction-file {output_dir}/prediction_hybrid_rank.txt \\")
            print(f"       --truth-file {truth_file}")
        else:
            print_section("NEXT STEPS - INFERENCE MODE")
            print("\n✓ Predictions generated successfully!")
            print("\n📤 Ready for submission:")
            print(f"   - WeightedMean: {output_dir}/prediction_weighted_rank.txt")
            print(f"   - Stacking:     {output_dir}/prediction_stacking_rank.txt")
            print(f"   - Hybrid:       {output_dir}/prediction_hybrid_rank.txt")
            print("\n💡 Pick the best method based on validation results")
            print("   (In this demo, use results from --mode eval to choose)")
        
    else:
        print_section("❌ FAILED!")
        print(f"Exit code: {result.returncode}")


if __name__ == '__main__':
    main()
