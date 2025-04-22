#!/usr/bin/env python3
"""
Cell Type Classifier Training Script for Tabnet

This script provides a flexible framework for training a Tabnet cell type classifier 
using PyTorch Lightning and a custom estimator. It supports comprehensive configuration 
via command-line arguments.
"""

import os
import argparse
import sys
import json
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.model_summary import ModelSummary

# ...existing code for path adjustments...
os.chdir('../scTab')
sys.path.append('.')
sys.path.append('../model_training')

from cellnet.estimators import EstimatorCellTypeClassifier
from train_utils import setup_environment, create_paths, configure_trainer, save_hparams, build_model_name

def parse_arguments():
    parser = argparse.ArgumentParser(description='Tabnet Classifier Training')
    # Data and label configuration
    parser.add_argument('--data_path', type=str, default='/data/sebacultrera/merlin_cxg_2023_05_15_sf-log1p')
    parser.add_argument('--class_weights_filename', type=str, default='class_weights.npy')
    # Tabnet-specific parameters
    parser.add_argument('--lambda_sparse', type=float, default=1e-5)
    parser.add_argument('--n_d', type=int, default=128)
    parser.add_argument('--n_a', type=int, default=64)
    parser.add_argument('--n_steps', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=1.3)
    parser.add_argument('--n_independent', type=int, default=5)
    parser.add_argument('--n_shared', type=int, default=3)
    parser.add_argument('--virtual_batch_size', type=int, default=256)
    parser.add_argument('--mask_type', type=str, default='entmax')
    parser.add_argument('--augment_training_data', type=lambda x: x.lower() in ['true', '1', 'yes'], default=True)
    parser.add_argument('--use_hierarchical_loss', type=lambda x: x.lower() in ['true', '1', 'yes'], default=False, help='Use hierarchical loss for training')
    # Optimizer and training parameters
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr_scheduler_step_size', type=int, default=1)
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=1)
    # Logging and checkpointing
    parser.add_argument('--log_dir', type=str, default='/data/sebacultrera/merlin_cxg_2023_05_15_sf-log1p/tb_logs')
    parser.add_argument('--model_name', type=str, default='cxg_2023_05_15_tabnet')
    parser.add_argument('--checkpoint_interval', type=int, default=1)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    # New argument for save_top_k
    parser.add_argument('--save_top_k', type=int, default=2, help='Number of top models to save')
    return parser.parse_args()

def main():
    args = parse_arguments()
    setup_environment(args.seed)
    
    # Build model name with optional hierarchical flag
    model_name = build_model_name(args, args.model_name)
    checkpoint_path, logs_path = create_paths(args.log_dir, model_name)
    
    estim = EstimatorCellTypeClassifier(args.data_path)
    
    estim.init_datamodule(batch_size=args.batch_size, sub_sample_frac=1.0)
    
    # Use 200 log steps and checkpoint_interval from args
    configure_trainer(estim, args, checkpoint_path, logs_path,
                        checkpoint_interval=args.checkpoint_interval,
                        log_every_n_steps=200)
    
    save_hparams(estim, args)
    
    estim.init_model(model_type='tabnet', model_kwargs={
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'optimizer': torch.optim.AdamW,
        'lr_scheduler': torch.optim.lr_scheduler.StepLR,
        'lr_scheduler_kwargs': {
            'step_size': args.lr_scheduler_step_size,
            'gamma': args.lr_scheduler_gamma,
            'verbose': True
        },
        'lambda_sparse': args.lambda_sparse,
        'n_d': args.n_d,
        'n_a': args.n_a,
        'n_steps': args.n_steps,
        'gamma': args.gamma,
        'n_independent': args.n_independent,
        'n_shared': args.n_shared,
        'virtual_batch_size': args.virtual_batch_size,
        'mask_type': args.mask_type,
        'augment_training_data': args.augment_training_data,
        'use_hierarchical_loss': args.use_hierarchical_loss
    })
    
    print(ModelSummary(estim.model))
    estim.train()

if __name__ == '__main__':
    main()