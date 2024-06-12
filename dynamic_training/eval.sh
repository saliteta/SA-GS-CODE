#!/bin/bash

# One need to activate nerfstudio environment

Model_Folder=(
    /home/xiongbutian/workspace/SAM-GS/output_alpha_constrain_eval/SZTU_COLMAP
)

# Define base folder

RENDER="python render.py -m"
EVAL="python metrics.py -m"

for i in "${!Model_Folder[@]}"; do
    # Check if it is a directory
    item="${Model_Folder[$i]}"
    if [ -d "$item" ]; then
    CUDAVISIBLE_DEVICES=7 $RENDER "${item}/" --skip_train
    CUDAVISIBLE_DEVICES=7 $EVAL "${item}/" 

    fi
done

