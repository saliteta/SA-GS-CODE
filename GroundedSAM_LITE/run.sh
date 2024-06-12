#!/bin/bash

# Base command for training
BASE_CMD="python -W ignore grounded_sam_demo_all.py \
--config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
--grounded_checkpoint /home/xiongbutian/workspace/GroundingSAM_Fast/groundingdino_swint_ogc.pth \
--sam_checkpoint /home/xiongbutian/workspace/GroundingSAM_Fast/sam_vit_h_4b8939.pth \
--device cuda \
--debugging True"

# Define the datasets
DATASETS=(
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
    output_npz="./${outpit}/${DATASET_NAME}/npz"
    output_img="./${outpit}/${DATASET_NAME}/img"
    
    # Ensure directories exist
    mkdir -p $output_npz
    mkdir -p $output_img

    # Log the activity
    echo "Using dataset at: ${DATASETS[$i]}"
    echo "Saving log file at: ${LOG_FILE_NAME}"
    echo "Saving npz file masks at: ${output_npz}"
    echo "Saving images at: ${output_img}"
    
    # Run the model with the specific GPU
    CUDA_VISIBLE_DEVICES=$(($i + 4)) $BASE_CMD --image_dir "${DATASETS[$i]}/images/" -o $output/$DATASET_NAME --prompts_file prompt.txt > "$LOG_FILE_NAME" 2>&1 &
done

# Wait for all background jobs to complete
wait

# Additional commands can go here if needed after all training is complete