import argparse
import sys
import os
os.chdir('../scTab')
sys.path.append('.')
sys.path.append('../model_evaluation')
import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import correct_labels
import torch
import yaml
from cellnet.estimators import EstimatorCellTypeClassifier
import lightning.pytorch as pl
import seaborn as sns
from cellnet.models import TabnetClassifier, LinearClassifier, MLPClassifier

SUBSET_INDICES = np.array([0, 3, 5, 10, 14, 17, 18, 19, 20, 21, 22, 24, 34,
    35, 36, 40, 43, 46, 47, 49, 50, 52, 53, 54, 59, 60,
    63, 64, 65, 68, 70, 77, 78, 79, 80, 82, 83, 86, 90,
    92, 94, 96, 99, 100, 101, 104, 106, 107, 108, 109, 113, 114,
    116, 117, 118, 119, 122, 123, 124, 125, 127, 128, 129, 130, 131,
    132, 133, 134, 135, 136, 138, 140, 141, 144, 145, 154, 155, 161,
    162, 163])

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate cell type classification models on scTab test set')
    
    # Data paths
    parser.add_argument('--data_root', type=str, required=True,
                      help='Root path to data')
    parser.add_argument('--cell_type_mapping_file', type=str, required=True,
                      help='Path to cell_type.parquet')
    parser.add_argument('--cell_type_hierarchy_file', type=str, required=True,
                      help='Path to child_matrix.npy')
    
    # Model paths and configuration
    parser.add_argument('--model_type', type=str, required=True,
                      choices=['tabnet', 'linear', 'mlp'],
                      help='Type of model to evaluate')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--hparams_file', type=str,
                      help='Path to hyperparameters file')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Directory to save evaluation results')
    
    return parser.parse_args()

def filter_by_cell_types(y_true, y_pred, indices):
    """Filter predictions and true labels for specific cell types."""
    mask = np.isin(y_true, indices)
    return y_true[mask], y_pred[mask]

def evaluate_model(estimator, cell_type_hierarchy_file):
    """Evaluate model performance using estimator's test dataloader."""
    print("\nEvaluating model...")
    
    # Get predictions using estimator
    probas = estimator.predict(estimator.datamodule.test_dataloader())
    
    # Move probabilities to CPU for post-processing
    if isinstance(probas, torch.Tensor):
        probas = probas.cpu().numpy()
    
    y_pred = np.argmax(probas, axis=1)
    
    # Get ground truth from test dataloader
    y_true = dd.read_parquet(
        os.path.join(estimator.data_path, 'test'), 
        columns='cell_type'
    ).compute().to_numpy()
    
    # Correct predictions using cell type hierarchy
    cell_type_hierarchy = np.load(cell_type_hierarchy_file)
    y_pred_corr = correct_labels(y_true, y_pred, cell_type_hierarchy)
    
    # Compute metrics for all cell types
    from sklearn.metrics import classification_report
    clf_report = pd.DataFrame(
        classification_report(y_true, y_pred_corr, output_dict=True)
    ).T
    
    clf_report_subset = pd.DataFrame(
        classification_report(y_true, y_pred_corr, output_dict=True, labels=SUBSET_INDICES)
    ).T

    return (
        clf_report.iloc[-3:].copy(),
        clf_report.iloc[:-3].copy(),
        clf_report_subset.iloc[-3:].copy()
    )

def save_results(clf_report_overall, clf_report_per_class, clf_report_subset, estimator, args):
    """Save evaluation results."""
    print("\nSaving results...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save overall metrics
    overall_path = os.path.join(args.output_dir, f'{args.model_type}_overall_metrics.csv')
    clf_report_overall.to_csv(overall_path)
    print(f"Overall metrics saved to: {overall_path}")
    print("\nOverall Results:")
    print(clf_report_overall)
    
    # Save subset metrics
    subset_path = os.path.join(args.output_dir, f'{args.model_type}_subset_metrics.csv')
    clf_report_subset.to_csv(subset_path)
    print(f"\nSubset metrics saved to: {subset_path}")
    print("\nSubset Results:")
    print(clf_report_subset)
    
    # Save per-class metrics
    per_class_path = os.path.join(args.output_dir, f'{args.model_type}_per_class_metrics.csv')
    clf_report_per_class.to_csv(per_class_path)
    print(f"Per-class metrics saved to: {per_class_path}")
    
    # Generate and save visualization
    plt.figure(figsize=(20, 3))
    cell_type_mapping = pd.read_parquet(args.cell_type_mapping_file)
    ax = plt.gca()
    
    # Create bar plot
    sns.barplot(
        data=(
            clf_report_per_class
            .assign(
                cell_type=lambda df: df.index.to_series().astype(int).replace(cell_type_mapping['label'].to_dict())
            )
            .sort_values('f1-score', ascending=False)
        ),
        x='cell_type',
        y='f1-score',
        color='#1f77b4',
        ax=ax
    )
    
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_xlabel('')
    ax.set_title(f'{args.model_type.capitalize()} Performance by Cell Type')
    ax.xaxis.set_tick_params(labelsize='small')
    
    plot_path = os.path.join(args.output_dir, f'{args.model_type}_performance_plot.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Performance plot saved to: {plot_path}")

def main():
    """Main execution function."""
    args = parse_args()
    
    # Initialize estimator and model
    print("\nInitializing model and data...")
    estim = EstimatorCellTypeClassifier(args.data_root)
    estim.init_datamodule(batch_size=2048)
    
    # Load appropriate model based on model type
    if args.model_type == 'tabnet':
        estim.model = TabnetClassifier.load_from_checkpoint(
            args.checkpoint_path, 
            **estim.get_fixed_model_params('tabnet')
        )
    elif args.model_type == 'linear':
        estim.model = LinearClassifier.load_from_checkpoint(
            args.checkpoint_path,
            **estim.get_fixed_model_params('linear')
        )
    elif args.model_type == 'mlp':
        estim.model = MLPClassifier.load_from_checkpoint(
            args.checkpoint_path,
            **estim.get_fixed_model_params('mlp')
        )
    
    # Use GPU for model evaluation
    gpu_available = torch.cuda.is_available()
    device = 'gpu' if gpu_available else 'cpu'
    print(f"Using {device} for model evaluation")
    
    # Configure trainer to use GPU
    estim.trainer = pl.Trainer(
        logger=[],
        accelerator=device,
        devices=1,
    )
    
    # Move model to appropriate device
    if gpu_available:
        estim.model = estim.model.cuda()
    
    # Evaluate model
    clf_report_overall, clf_report_per_class, clf_report_subset = evaluate_model(
        estim, args.cell_type_hierarchy_file
    )
    
    # Save results
    save_results(clf_report_overall, clf_report_per_class, clf_report_subset, estim, args)

if __name__ == '__main__':
    main()
