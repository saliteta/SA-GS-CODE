#!/bin/bash

# Define the base folder
Model_Folder=(
    /home/xiongbutian/workspace/SAM-GS/output_alpha_constrain_eval/SZTU_COLMAP/
)


# Commands for rendering and evaluating
RENDER="python render.py -m"
EVAL="python metrics.py -m"

# Create a directory for logs, if it doesn't already exist
mkdir -p ../logs_eval/out_alpha_constrain/

# Find the number of GPUs available
NUM_GPUS=$(nvidia-smi -L | wc -l)

# Initialize a GPU counter
gpu_id=7

# Iterate over all directories in Model_Folder
for i in "${!Model_Folder[@]}"; do
    item=${Model_Folder[$i]}
    echo item
    if [ -d "$item" ]; then  # Check if it is a directory
        model_folder_name=$(basename "$item")
        LOG_FILE_NAME="../logs_eval/out_alpha_constrain/${model_folder_name}.txt"
        echo "Logging to ${LOG_FILE_NAME}"

        # Run both commands in a subshell to share the same log file
        (
            CUDA_VISIBLE_DEVICES=$gpu_id $RENDER "${item}/" --skip_train
            CUDA_VISIBLE_DEVICES=$gpu_id $EVAL "${item}/"
        ) > "$LOG_FILE_NAME" 2>&1 

        # Increment the gpu_id and wrap around if necessary
    fi
done

# Wait for all background processes to finish
wait
