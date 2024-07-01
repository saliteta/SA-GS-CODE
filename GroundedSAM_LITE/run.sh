#!/bin/bash

# Base command for training
BASE_CMD="python -W ignore grounded_sam_demo_all.py \
--config <Where your Grounding DINO model config file is located> xxx/GroundingDINO_SwinT_OGC.py \
--grounded_checkpoint <Where your GroundingDINO model is located> xxx/groundingdino_swint_ogc.pth \
--sam_checkpoint <Where you SAM model is located> sam_vit_h_4b8939.pth \
--device cuda \
--debugging <When set to true, it will save some segmented picture out here> True"

# Define the datasets

DATASETS=(
    # If you have multiple COLMAP dataset, you can put it here. 
    # This will allow you run multiple experiment in parallel
    /data/grocery_store/10F
)



# Current date for logs
MISSION_START_DATE=$(date +%Y%m%d)

# Create directories for logs and output
output=./output
mkdir -p ./logs
mkdir -p $output
# Loop through the datasets and assign each to a GPU, adding port and output directory
for i in "${!DATASETS[@]}"; do
    DATASET_NAME=$(basename "${DATASETS[$i]}")
    LOG_FILE_NAME="./logs/${MISSION_START_DATE}_${DATASET_NAME}.txt"
    output_npz="${output}/${DATASET_NAME}/npz"
    output_img="${output}/${DATASET_NAME}/img"
    
    # Ensure directories exist
    mkdir -p $output_npz
    mkdir -p $output_img

    # Log the activity
    echo "Using dataset at: ${DATASETS[$i]}"
    echo "Saving log file at: ${LOG_FILE_NAME}"
    echo "Saving npz file masks at: ${output_npz}"
    echo "Saving images at: ${output_img}"
    
    # Run the model with the specific GPU
    CUDA_VISIBLE_DEVICES=$(($i)) $BASE_CMD --image_dir "${DATASETS[$i]}/images/" -o $output/$DATASET_NAME --prompts_file prompt.txt > "$LOG_FILE_NAME" 2>&1 &
done

# Wait for all background jobs to complete
wait

# Additional commands can go here if needed after all training is complete
