#!/bin/bash

# One need to activate nerfstudio environment

# Define base folder
Model_Folder="/home/xiongbutian/workspace/SAM-GS/output_alpha_constrain"
output_dir="/home/xiongbutian/workspace/SAM-GS/extracted_point_cloud/probability_extraction/alpha_constrain"
# Define the model path relative to each item folder
mkdir -p $output_dir

CMD="python point_cloud_extraction.py"

for item in "$Model_Folder"/*; do
    # Check if it is a directory
    echo $item
    if [ -d "$item" ]; then

    plyname="$(basename "$item").ply"
    $CMD  --gs_model "${item}/point_cloud/iteration_30000/point_cloud.ply" --store_location $output_dir/$plyname 
    fi
done