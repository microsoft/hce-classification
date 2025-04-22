#!/bin/bash

# Common data paths
DATA_ROOT="/data/sebacultrera/merlin_cxg_2023_05_15_sf-log1p"

# Function to evaluate a single checkpoint
evaluate_checkpoint() {
    CHECKPOINT_PATH="$1"
    
    # Dynamically extract log and model folders from the checkpoint file path
    REL_PATH=${CHECKPOINT_PATH#${DATA_ROOT}/}
    LOG_DIR=$(echo "$REL_PATH" | cut -d'/' -f1)
    MODEL_NAME=$(echo "$REL_PATH" | cut -d'/' -f2)
    DEFAULT_DIR=$(echo "$REL_PATH" | cut -d'/' -f3)
    VERSION_DIR=$(echo "$REL_PATH" | cut -d'/' -f4)
    MODEL_PATH="${DATA_ROOT}/${LOG_DIR}/${MODEL_NAME}/${DEFAULT_DIR}/${VERSION_DIR}"
    
    CHECKPOINT_NAME=$(basename "$CHECKPOINT_PATH" .ckpt)
    
    # Determine model type based on checkpoint path
    if [[ $CHECKPOINT_PATH == *"mlp"* ]]; then
        MODEL_TYPE="mlp"
    elif [[ $CHECKPOINT_PATH == *"tabnet"* ]]; then
        MODEL_TYPE="tabnet"
    elif [[ $CHECKPOINT_PATH == *"linear"* ]]; then
        MODEL_TYPE="linear"
    else
        echo "Unknown model type in path: $CHECKPOINT_PATH"
        return 1
    fi
    
    HPARAMS_PATH="${MODEL_PATH}/hparams.yaml"
    OUTPUT_DIR="${MODEL_PATH}/checkpoints/${CHECKPOINT_NAME}"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Run evaluation
    python model_evaluation.py \
        --dataset_ids "diff_2023-05-15" \
        --features_file "/home/sebacultrera/label_smoothing_celltype/label_smoothing_celltype/scTab/notebooks/store_creation/features.parquet" \
        --var_file "${DATA_ROOT}/var.parquet" \
        --cell_type_mapping_file "${DATA_ROOT}/categorical_lookup/cell_type.parquet" \
        --cell_type_hierarchy_file "${DATA_ROOT}/cell_type_hierarchy/child_matrix.npy" \
        --model_type "${MODEL_TYPE}" \
        --checkpoint_path "${CHECKPOINT_PATH}" \
        --hparams_file "${HPARAMS_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --output_root "${DATA_ROOT}" \
        --census_version "2023-12-15" \
        > "${OUTPUT_DIR}/eval.out" 2> "${OUTPUT_DIR}/eval.err"
}

export -f evaluate_checkpoint

find_checkpoints() {
    local dir="$1"
    find "$dir" -name "*.ckpt"
}

# Check if checkpoint list file is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <checkpoint_list_file>"
    exit 1
fi

CHECKPOINT_LIST="$1"
MAX_PARALLEL=8  # Maximum number of parallel processes
running=0       # Counter for running processes

# Read checkpoints and run evaluations
while IFS= read -r path; do
    # Skip empty lines
    [ -z "$path" ] && continue
    
    if [ -d "$path" ]; then
        # If path is a directory, find all checkpoints
        while IFS= read -r checkpoint; do
            # Wait if we've reached max parallel processes
            if [ $running -ge $MAX_PARALLEL ]; then
                wait -n
                running=$((running - 1))
            fi
            
            evaluate_checkpoint "$checkpoint" &
            running=$((running + 1))
        done < <(find_checkpoints "$path")
    else
        # Handle single checkpoint file
        if [ $running -ge $MAX_PARALLEL ]; then
            wait -n
            running=$((running - 1))
        fi
        
        evaluate_checkpoint "$path" &
        running=$((running + 1))
    fi
done < "$CHECKPOINT_LIST"

# Wait for remaining processes to finish
wait
