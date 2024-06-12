#!/bin/bash

# Define the base command
CMD="python train.py --eval"

# Define the base port
BASE_PORT=7900

# Define the datasets
DATASETS=(
    /data/grocery_store/10F
)

export CUDA_VISIBLE_DEVICES=4,5
# Get the current date in YYYYMMDD format
MISSION_START_DATE=$(date +%Y%m%d)

# Create a logs directory if it doesn't already exist
mkdir -p ./logs
mkdir -p ./output_alpha_constrain_eval

# Loop through the datasets and assign each to a GPU, add port and output directory
for i in "${!DATASETS[@]}"; do
    # Calculate the port for the current dataset
    PORT=$(($BASE_PORT + i))
    echo $PORT
    
    # Extract the dataset name for the output directory and log file name
    DATASET_NAME=$(basename "${DATASETS[$i]}")
    
    # Construct the log file name within the logs folder
    LOG_FILE_NAME="./logs/${MISSION_START_DATE}_${DATASET_NAME}.txt"
    COMPLEXITY="../${DATASET_NAME}_geometric_complexity.csv"
    
    # Construct and execute the command
    echo "this is the dataset name ${DATASETS[$i]}"
    echo "this is the complexity name I want to load name ${COMPLEXITY}"
    CUDA_VISIBLE_DEVICES=4 $CMD -s "${DATASETS[$i]}" \
    --data_device cpu \
    --port $PORT \
    -m "../output/${DATASET_NAME}" \
    --mask_path "/home/xiongbutian/workspace/SA-GS/GroundedSAM_LITE/output/10F/npz"\
    --perplexity_path $COMPLEXITY > "$LOG_FILE_NAME" 2>&1 &
done

wait
