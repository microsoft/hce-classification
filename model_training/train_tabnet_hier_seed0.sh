#!/bin/bash
python train_tabnet.py \
    --data_path /data/sebacultrera/merlin_cxg_2023_05_15_sf-log1p \
    --class_weights_filename class_weights.npy \
    --lambda_sparse 1e-5 \
    --n_d 128 \
    --n_a 64 \
    --n_steps 1 \
    --gamma 1.3 \
    --n_independent 5 \
    --n_shared 3 \
    --virtual_batch_size 256 \
    --mask_type entmax \
    --augment_training_data True \
    --lr 0.005 \
    --weight_decay 0.05 \
    --batch_size 2048 \
    --epochs 50 \
    --seed 0 \
    --log_dir /data/sebacultrera/merlin_cxg_2023_05_15_sf-log1p/tb_logs \
    --model_name cxg_2023_05_15_tabnet \
    --lr_scheduler_step_size 1 \
    --lr_scheduler_gamma 0.9 \
    --checkpoint_interval 1 \
    --check_val_every_n_epoch 1 \
    --use_hierarchical_loss True \
    --save_top_k 1 \
    > logs/train_tabnet_hier_seed0.out 2> logs/train_tabnet_hier_seed0.err
