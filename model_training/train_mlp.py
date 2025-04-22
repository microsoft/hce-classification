#!/usr/bin/env python3
"""
Cell Type Classifier Training Script

This script provides a flexible framework for training a linear cell type classifier 
using PyTorch Lightning and a custom estimator. It supports comprehensive 
configuration via command-line arguments and enables reproducible machine learning 
experiments in single-cell data analysis.

Key Features:
- Configurable model hyperparameters
- Flexible trainer configuration
- Learning rate scheduling
- TensorBoard logging
"""

import os
import argparse
import sys
import numpy as np
import torch
import seaborn as sns

from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (
    ModelCheckpoint, 
    LearningRateMonitor, 
    TQDMProgressBar
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.model_summary import ModelSummary

# Ensure local package is importable
# sys.path.append('..')
# sys.path.append('.')

os.chdir('../scTab')
sys.path.append('.')
sys.path.append('../model_training')

from cellnet.estimators import EstimatorCellTypeClassifier
from train_utils import setup_environment, build_model_name, create_paths, configure_trainer, save_hparams

def parse_arguments():
    """
    Parse command-line arguments for comprehensive model and training configuration.
    
    Returns:
        argparse.Namespace: Parsed configuration arguments
    """
    parser = argparse.ArgumentParser(description='Cell Type Classifier Training')
    
    # Data Configuration
    parser.add_argument('--data_path', type=str, 
                        default='/data/sebacultrera/merlin_cxg_2023_05_15_sf-log1p',
                        help='Path to input data')
    parser.add_argument('--class_weights_filename', type=str, default='class_weights.npy',
                        help='Filename for class weights')
    
    # Model Configuration
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay for regularization')
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--n_hidden', default=8, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--augment_training_data', default=True, type=lambda x: x.lower() in ['true', '1', '1.'])
    parser.add_argument('--use_hierarchical_loss', default=False, type=lambda x: x.lower() in ['true', '1', '1.'])
    
    # Training Configuration
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Training batch size')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum number of training epochs')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for reproducibility')
    
    # Logging and Checkpointing
    parser.add_argument('--log_dir', type=str, default='./tb_logs',
                        help='Base directory for TensorBoard logs')
    parser.add_argument('--model_name', type=str, default='cxg_2023_05_15_mlp',
                        help='Name for logging and checkpointing')
    
    # Learning Rate Scheduler Configuration
    parser.add_argument('--lr_scheduler_step_size', type=int, default=1,
                        help='Step size for learning rate scheduler')
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.9,
                        help='Gamma decay for learning rate scheduler')
    # Expose gradient clipping value as a parameter
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                        help='Gradient clipping value')
    # New argument for save_top_k
    parser.add_argument('--save_top_k', type=int, default=2, help='Number of top models to save')
    
    return parser.parse_args()

def main():
    """
    Main training routine for the cell type classifier.
    Configures and executes the entire training pipeline.
    """
    # Parse command-line arguments
    args = parse_arguments()
    setup_environment(args.seed)
    # Build model name with optional hierarchical flag
    model_name = build_model_name(args, args.model_name)
    checkpoint_path, logs_path = create_paths(args.log_dir, model_name)
    
    estim = EstimatorCellTypeClassifier(args.data_path, class_weights_filename=args.class_weights_filename)
    
    estim.init_datamodule(batch_size=args.batch_size)
    
    configure_trainer(estim, args, checkpoint_path, logs_path,
                        checkpoint_interval=1, log_every_n_steps=100)
    
    save_hparams(estim, args)
    
    # Initialize model
    estim.init_model(model_type='mlp', model_kwargs={
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'optimizer': torch.optim.AdamW,
        'lr_scheduler': torch.optim.lr_scheduler.StepLR,
        'lr_scheduler_kwargs': {
            'step_size': args.lr_scheduler_step_size,
            'gamma': args.lr_scheduler_gamma,
            'verbose': True
        },
        'hidden_size': args.hidden_size,
        'n_hidden': args.n_hidden,
        'dropout': args.dropout,
        'augment_training_data': args.augment_training_data,
        'use_hierarchical_loss': args.use_hierarchical_loss
    })
    
    # Print model summary and start training
    print(ModelSummary(estim.model))
    estim.train()

if __name__ == '__main__':
    main()