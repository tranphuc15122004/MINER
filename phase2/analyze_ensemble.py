"""
Script để visualize và analyze saved ensemble models

Sử dụng để:
- Xem training history (Optuna trials, fold metrics)
- Plot optimization curves
- Generate báo cáo cho paper/thesis
"""
import argparse
import json
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd


def analyze_weighted_mean(model_dir: str, output_dir: str):
    """Analyze WeightedMean ensemble"""
    print("\n" + "="*80)
    print("WEIGHTED MEAN ENSEMBLE ANALYSIS")
    print("="*80)
    
    # Load metadata
    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("\n[Training Configuration]")
    print(f"Number of trials: {metadata['n_trials']}")
    print(f"Sampling rate: {metadata['sampling_rate']}")
    print(f"Number of models: {metadata['num_models']}")
    print(f"Timestamp: {metadata['timestamp']}")
    
    print("\n[Best Results]")
    print(f"Best AUC: {metadata['best_auc']:.4f}")
    print(f"Best weights:")
    for model, weight in metadata['best_weights'].items():
        print(f"  {model}: {weight:.4f}")
    
    # Load training history
    history_path = os.path.join(model_dir, 'training_history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        print(f"\n[Training History]")
        print(f"Total trials: {len(history)}")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(history)
        print(f"Best trial: #{df.loc[df['value'].idxmax(), 'trial']}")
        print(f"Mean AUC: {df['value'].mean():.4f}")
        print(f"Std AUC: {df['value'].std():.4f}")
        
        # Plot optimization history
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['trial'], df['value'], alpha=0.6, label='Trial AUC')
        plt.plot(df['trial'], df['value'].cummax(), 'r-', linewidth=2, label='Best AUC')
        plt.xlabel('Trial')
        plt.ylabel('AUC')
        plt.title('Optuna Optimization History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'optuna_history.png')
        plt.savefig(plot_path, dpi=300)
        print(f"\nSaved plot: {plot_path}")
        
        # Weight distribution across trials
        if len(history) > 0 and 'params' in history[0]:
            weight_data = {k: [] for k in history[0]['params'].keys()}
            for trial in history:
                for k, v in trial['params'].items():
                    weight_data[k].append(v)
            
            plt.figure(figsize=(12, 6))
            for i, (model, weights) in enumerate(weight_data.items()):
                plt.subplot(1, len(weight_data), i+1)
                plt.hist(weights, bins=20, alpha=0.7)
                plt.xlabel('Weight')
                plt.ylabel('Frequency')
                plt.title(f'{model}')
                plt.axvline(metadata['best_weights'][model], color='r', 
                           linestyle='--', label='Best')
                plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'weight_distribution.png')
            plt.savefig(plot_path, dpi=300)
            print(f"Saved plot: {plot_path}")
    
    # Load Optuna study for more detailed analysis
    study_path = os.path.join(model_dir, 'optuna_study.pkl')
    if os.path.exists(study_path):
        import optuna
        with open(study_path, 'rb') as f:
            study = pickle.load(f)
        
        # Plot parameter importances
        try:
            fig = optuna.visualization.plot_param_importances(study)
            plot_path = os.path.join(output_dir, 'param_importances.png')
            fig.write_image(plot_path)
            print(f"Saved plot: {plot_path}")
        except:
            print("Note: Install plotly and kaleido for advanced visualizations")
    
    # Generate LaTeX table for paper
    latex_table = generate_weights_latex_table(metadata['best_weights'], metadata['best_auc'])
    latex_path = os.path.join(output_dir, 'weights_table.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"\nGenerated LaTeX table: {latex_path}")


def analyze_stacking(model_dir: str, output_dir: str):
    """Analyze Stacking ensemble"""
    print("\n" + "="*80)
    print("STACKING ENSEMBLE ANALYSIS")
    print("="*80)
    
    # Load metadata
    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("\n[Training Configuration]")
    print(f"Number of folds: {metadata['n_folds']}")
    print(f"Sampling rate: {metadata['sampling_rate']}")
    print(f"Number of features: {metadata['num_features']}")
    print(f"Timestamp: {metadata['timestamp']}")
    
    print("\n[LightGBM Parameters]")
    for key, value in metadata['lgb_params'].items():
        print(f"  {key}: {value}")
    
    print("\n[Results]")
    print(f"OOF AUC: {metadata['oof_auc']:.4f}")
    
    print("\n[Fold Metrics]")
    for fold_metric in metadata['fold_metrics']:
        print(f"Fold {fold_metric['fold']}:")
        print(f"  Best iteration: {fold_metric['best_iteration']}")
        print(f"  Best score: {fold_metric['best_score']}")
    
    # Plot fold performance
    os.makedirs(output_dir, exist_ok=True)
    
    df_folds = pd.DataFrame(metadata['fold_metrics'])
    
    plt.figure(figsize=(10, 6))
    plt.bar(df_folds['fold'], [m['valid_0']['ndcg@10'] for m in df_folds['best_score']])
    plt.xlabel('Fold')
    plt.ylabel('NDCG@10')
    plt.title('LightGBM Fold Performance')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'fold_performance.png')
    plt.savefig(plot_path, dpi=300)
    print(f"\nSaved plot: {plot_path}")
    
    # Feature importance (if models exist)
    try:
        import lightgbm as lgb
        
        # Load first fold model for feature importance
        model_path = os.path.join(model_dir, 'fold_1.txt')
        if os.path.exists(model_path):
            model = lgb.Booster(model_file=model_path)
            
            # Get feature importance
            importance = model.feature_importance(importance_type='gain')
            feature_names = model.feature_name()
            
            # Plot top 20 features
            df_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(20)
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(df_importance)), df_importance['importance'])
            plt.yticks(range(len(df_importance)), df_importance['feature'])
            plt.xlabel('Importance (Gain)')
            plt.ylabel('Feature')
            plt.title('Top 20 Feature Importances (Fold 1)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'feature_importance.png')
            plt.savefig(plot_path, dpi=300)
            print(f"Saved plot: {plot_path}")
            
            # Save feature importance to CSV
            csv_path = os.path.join(output_dir, 'feature_importance.csv')
            pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).to_csv(csv_path, index=False)
            print(f"Saved CSV: {csv_path}")
    except Exception as e:
        print(f"Could not analyze feature importance: {e}")
    
    # Generate LaTeX table
    latex_table = generate_stacking_latex_table(metadata)
    latex_path = os.path.join(output_dir, 'stacking_results.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"\nGenerated LaTeX table: {latex_path}")


def generate_weights_latex_table(weights: dict, auc: float) -> str:
    """Generate LaTeX table for weights"""
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Optimized Ensemble Weights (Optuna)}",
        "\\begin{tabular}{lc}",
        "\\hline",
        "Model & Weight \\\\",
        "\\hline"
    ]
    
    for model, weight in weights.items():
        lines.append(f"{model.replace('_', '\\_')} & {weight:.4f} \\\\")
    
    lines.extend([
        "\\hline",
        f"\\multicolumn{{2}}{{c}}{{Validation AUC: {auc:.4f}}} \\\\",
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)


def generate_stacking_latex_table(metadata: dict) -> str:
    """Generate LaTeX table for stacking results"""
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{LightGBM Stacking Results}",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Fold & Best Iteration & NDCG@5 & NDCG@10 \\\\",
        "\\hline"
    ]
    
    for fold_metric in metadata['fold_metrics']:
        fold = fold_metric['fold']
        iter = fold_metric['best_iteration']
        ndcg5 = fold_metric['best_score']['valid_0']['ndcg@5']
        ndcg10 = fold_metric['best_score']['valid_0']['ndcg@10']
        lines.append(f"{fold} & {iter} & {ndcg5:.4f} & {ndcg10:.4f} \\\\")
    
    lines.extend([
        "\\hline",
        f"\\multicolumn{{4}}{{c}}{{OOF AUC: {metadata['oof_auc']:.4f}}} \\\\",
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze saved ensemble models")
    parser.add_argument('--model-dir', required=True,
                        help='Directory containing saved model')
    parser.add_argument('--output-dir', default='phase2/analysis',
                        help='Output directory for plots and reports')
    
    args = parser.parse_args()
    
    # Detect model type
    metadata_path = os.path.join(args.model_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    model_type = metadata['method']
    
    # Analyze based on type
    if model_type == 'WeightedMean':
        analyze_weighted_mean(args.model_dir, args.output_dir)
    elif model_type == 'StackingLightGBM':
        analyze_stacking(args.model_dir, args.output_dir)
    else:
        print(f"Unknown model type: {model_type}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
