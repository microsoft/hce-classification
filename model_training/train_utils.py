import os
import json
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np

def build_model_name(args, default_model_name):
    name = default_model_name
    if getattr(args, 'use_hierarchical_loss', False):
        name += '_hierarchical_loss'
    return name

def create_paths(log_dir, model_name):
    checkpoint_path = os.path.join(log_dir, model_name)
    logs_path = os.path.join(log_dir, model_name)
    os.makedirs(logs_path, exist_ok=True)
    return checkpoint_path, logs_path

def setup_environment(seed):
    seed_everything(seed)
    torch.set_float32_matmul_precision('high')

def get_common_callbacks(args, checkpoint_interval, monitor_f1_key='val_f1_macro', monitor_loss_key='val_loss'):
    return [
        TQDMProgressBar(refresh_rate=250),
        LearningRateMonitor(logging_interval='step'),
        # ModelCheckpoint(filename='last_{epoch}', every_n_epochs=checkpoint_interval, save_top_k=args.save_top_k),
        ModelCheckpoint(filename=f'{monitor_f1_key}_{{epoch}}_{{{monitor_f1_key}:.3f}}',
                        monitor=monitor_f1_key, mode='max', every_n_epochs=checkpoint_interval, save_top_k=args.save_top_k),
        # ModelCheckpoint(filename=f'{monitor_loss_key}_{{epoch}}_{{{monitor_loss_key}:.3f}}',
        #                 monitor=monitor_loss_key, mode='min', every_n_epochs=checkpoint_interval, save_top_k=args.save_top_k),
        EarlyStopping(monitor=monitor_f1_key, patience=10, mode='max'),
    ]

def configure_trainer(estim, args, checkpoint_path, logs_path, checkpoint_interval=1, log_every_n_steps=100,
                      monitor_f1_key='val_f1_macro', monitor_loss_key='val_loss'):
    callbacks = get_common_callbacks(args, checkpoint_interval, monitor_f1_key, monitor_loss_key)
    trainer_kwargs = {
        'max_epochs': getattr(args, 'epochs', getattr(args, 'max_epochs', 50)),
        'default_root_dir': checkpoint_path,
        'gradient_clip_val': getattr(args, 'gradient_clip_val', 1.0),
        'gradient_clip_algorithm': 'norm',
        'accelerator': 'gpu',
        'devices': 1,
        'num_sanity_val_steps': 0,
        'logger': [TensorBoardLogger(logs_path, name='default')],
        'log_every_n_steps': log_every_n_steps,
        'detect_anomaly': False,
        'enable_progress_bar': True,
        'enable_model_summary': False,
        'enable_checkpointing': True,
        'callbacks': callbacks,
    }
    estim.init_trainer(trainer_kwargs=trainer_kwargs)

def save_hparams(estim, args):
    hparams_dir = estim.trainer.log_dir
    os.makedirs(hparams_dir, exist_ok=True)
    with open(os.path.join(hparams_dir, "parameters.json"), "w") as f:
        json.dump(vars(args), f, indent=4)