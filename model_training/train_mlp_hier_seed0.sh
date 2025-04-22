#!/bin/bash
python train_mlp.py \
    --data_path /data/sebacultrera/merlin_cxg_2023_05_15_sf-log1p \
    --class_weights_filename class_weights.npy \
    --learning_rate 0.002 \
    --weight_decay 0.05 \
    --hidden_size 128 \
    --n_hidden 8 \
    --dropout 0.1 \
    --augment_training_data True \
    --use_hierarchical_loss True \
    --batch_size 2048 \
    --max_epochs 50 \
    --seed 0 \
    --log_dir /data/sebacultrera/merlin_cxg_2023_05_15_sf-log1p/tb_logs \
    --model_name cxg_2023_05_15_mlp \
    --lr_scheduler_step_size 1 \
    --lr_scheduler_gamma 0.9 \
    --gradient_clip_val 1.0 \
    --save_top_k 1 \
    > logs/train_mlp_hier_seed0.out 2> logs/train_mlp_hier_seed0.err
