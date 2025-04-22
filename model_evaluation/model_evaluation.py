import argparse
import sys
import os
sys.path.append('../scTab')
sys.path.append('../model_evaluation')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    data_preparation, 
    run_model, 
    print_clf_report_per_class
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate cell type classification models')
    
    # Data paths
    parser.add_argument('--dataset_ids', type=str, help='Dataset ID or "diff_2023-05-15"')
    parser.add_argument('--features_file', type=str, required=True,
                      help='Path to features.parquet')
    parser.add_argument('--var_file', type=str, required=True,
                      help='Path to var.parquet')
    parser.add_argument('--cell_type_mapping_file', type=str, required=True,
                      help='Path to cell_type.parquet')
    parser.add_argument('--cell_type_hierarchy_file', type=str, required=True,
                      help='Path to child_matrix.npy')
    
    # Model paths and configuration
    parser.add_argument('--model_type', type=str, required=True,
                      choices=['tabnet', 'linear', 'mlp', 'celltypist'],
                      help='Type of model to evaluate')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--hparams_file', type=str,
                      help='Path to hyperparameters file (not needed for CellTypist)')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Directory to save evaluation results')
    parser.add_argument('--census_version', type=str, default='2023-05-15',
                      help='CellXGene census version')
    parser.add_argument('--force_download', action='store_true',
                      help='Force re-download of data')
    
    # Add output root argument
    parser.add_argument('--output_root', type=str, required=True,
                      help='Root directory for storing AnnData chunks and results')
    
    return parser.parse_args()

def save_results(clf_report_overall, clf_report_per_class, y_probs, y_pred, y_true, metadata, cell_type_mapping, args):
    """Save evaluation results to files."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save overall metrics
    overall_path = os.path.join(args.output_dir, f'{args.model_type}_overall_metrics.csv')
    clf_report_overall.to_csv(overall_path)
    print(f"\nOverall metrics saved to: {overall_path}")
    print("\nOverall Results:")
    print(clf_report_overall)
    
    # Save per-class metrics
    per_class_path = os.path.join(args.output_dir, f'{args.model_type}_per_class_metrics.csv')
    clf_report_per_class.to_csv(per_class_path)
    print(f"\nPer-class metrics saved to: {per_class_path}")
    
    # Generate and save visualization
    plt.figure(figsize=(20, 10))
    print_clf_report_per_class(
        clf_report_per_class, 
        args.cell_type_mapping_file, 
        title=f'{args.model_type.capitalize()} Performance by Cell Type'
    )
    plot_path = os.path.join(args.output_dir, f'{args.model_type}_performance_plot.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\nPerformance plot saved to: {plot_path}")
    
    # Create and save detailed results dataframe
    print("\nCreating detailed results dataframe...")
    
    # Convert numeric indices to cell type labels
    cell_type_mapping_df = pd.read_parquet(args.cell_type_mapping_file)
    cell_type_mapping_dict = dict(zip(range(len(cell_type_mapping_df)), cell_type_mapping_df['label']))
    
    # Create the detailed results dataframe
    detailed_df = pd.DataFrame({
        'y_true': [cell_type_mapping_dict[idx] for idx in y_true],
        'y_pred': [cell_type_mapping_dict[idx] for idx in y_pred]
    })
    
    # Add other columns from metadata
    detailed_df = pd.concat([detailed_df, metadata.reset_index(drop=True)], axis=1)
    
    # Add probabilities as a column
    detailed_df['y_probs'] = list(y_probs)
    
    # Save the detailed results
    detailed_path = os.path.join(args.output_dir, f'{args.model_type}_detailed_results.parquet')
    detailed_df.to_parquet(detailed_path, index=False)
    print(f"\nDetailed results saved to: {detailed_path}")

def main():
    """Main execution function."""
    args = parse_args()
    
    print(f"\nPreparing data for {args.model_type} model evaluation...")
    output_folder, genes, cell_mapping = data_preparation(
        args.dataset_ids,
        args.features_file,
        args.var_file,
        args.cell_type_mapping_file,
        census_version=args.census_version,
        force_download=args.force_download,
        output_root=args.output_root
    )
    
    print(f"\nRunning evaluation for {args.model_type} model...")
    results = run_model(
        args.model_type,
        args.checkpoint_path,
        args.hparams_file,
        args.cell_type_hierarchy_file,
        genes,
        cell_mapping,
        output_folder
    )
    
    # Unpack results (now including probabilities and additional metadata)
    clf_report_overall, clf_report_per_class, y_probs, y_pred, y_true, metadata = results
    
    save_results(clf_report_overall, clf_report_per_class, y_probs, y_pred, y_true, metadata, cell_mapping, args)

if __name__ == '__main__':
    main()
