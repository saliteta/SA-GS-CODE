#!/bin/bash

# Define the base command

# Get the current date in YYYYMMDD format
MISSION_START_DATE=$(date +%Y%m%d)

# Create a logs directory if it doesn't already exist
mkdir -p logs

DATASETS=(
    #/data/GauUscene/GauU_Scene/SZTU_COLMAP/
    /data/grocery_store/10F/
    #/data/GauUscene/urbanscene3DrealScene/polytech
)

mask_folder=/home/xiongbutian/workspace/SA-GS/GroundedSAM_LITE/output/10F/npz

# Get the number of available GPUs
# GPU Counter to keep track of which GPU to use next

for i in "${!DATASETS[@]}"; do
    images_folder="${DATASETS[$i]}"
    if [ -d "$images_folder" ]; then # Check if it is a directory
        basename=$(basename "$images_folder")
        echo "This is the name of processed dataset: $basename"
        LOG_FILE_NAME="logs/${MISSION_START_DATE}_${basename}.txt"

        output_file="../${basename}_geometric_complexity.csv"
        
        
        # Run the process in the background
        python main.py --image_path $images_folder/images --mask_path $mask_folder --prompts_file ../prompt.txt --output_file $output_file > $LOG_FILE_NAME 2>&1 &
    fi
done

# Wait for all background processes to finish
wait
