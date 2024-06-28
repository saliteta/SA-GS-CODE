# SA-GS

## Environment Installation

- Clone Code

```bash
git clone --recursive https://github.com/saliteta/SA-GS-CODE.git
cd SA-GS-CODE
```

- Install Environment

```bash
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate SA-GS
```

## Pretrained Model

In our training process, we need to use GroundingDINO model and SAM model. We need to locate where the model is. For your referece. The model has the following name:

```bash
sam_vit_h_4b8939.pth
groundingdino_swint_ogc.pth
```

To download the model, one can go to the following link:

https://huggingface.co/spaces/abhishek/StableSAM/blob/main/sam_vit_h_4b8939.pth

https://huggingface.co/ShilongLiu/GroundingDINO/tree/main

## Dataset Preparation

To use GauUsceneV2 dataset, one can view our dataset webpage and follow the instruction to get full access to COLMAP dataset: 

https://saliteta.github.io/CUHKSZ_SMBU/

If one want to use their own COLMAP dataset, one can follow exactly what Vanilla Gaussian Splatting Dataset Procedure to construct their own dataset: 

https://github.com/graphdeco-inria/gaussian-splatting

## Prompt Text Preparation

One needs to write the prompt in a text file like this:

```bash
ground
building
vegetation
```

One can modify the prompt as what they want in their experiment. In our paper, we use the above prompts

## Running

- Mask Extraction

One can use our [run.sh](http://run.sh) in GrounedSAM_LITE as a start. Remember to modify this part:

```bash
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
```

- Geometric Complexity Calculation

One can use the shell file in geometric_measurement called [test.sh](http://test.sh) to calculate the geometric

Rememeber to modify this part to the dataset and correct data masks as one wants. We also provide multi-threading in our CPU calculation

```bash
DATASETS=(
    /data/grocery_store/10F/
)

mask_folders=(
    ../GroundedSAM_LITE/output/10F/npz
)

prompts=(
    ../prompt.txt
)

```

- Dynamic Training

If one want to measure the geometric difference using point cloud, then DO NOT add —eval in the shell file called run_in_parallel.sh

```bash
CMD="python train.py --eval" 
```

One needs to modify the dataset here:

```bash
# Define the datasets
DATASETS=(
    /data/grocery_store/10F
)
MASK_FOLDERS=(
    ../GroundedSAM_LITE/output/10F/npz
)
```

- Point Cloud Extraction

Using extract_point_cloud.sh or directly use the following command: 

```bash
python point_cloud_extraction.py <The Gaussian Model> xxx.ply --store_location <the place you want to store your point cloud>xx.ply
```

If one use our shell file, do not forget to modify the dataset name

## Evaluation

- Geometric Based Metrics

To measure the geometric metrics, and visualize the result, one can download cloud compare:

https://www.danielgm.net/cc/

And open the LiDAR data and ply file we trained. After aligning LiDAR (Use the transformation matrix GauUsceneV2 provided) we crop the size of our point cloud the same as LiDAR. And then use CD difference. Detailed comparison methods and transformation is here:

https://www.cloudcompare.org/doc/wiki/index.php/Apply_Transformation

[Distances Computation - CloudCompareWiki](https://www.cloudcompare.org/doc/wiki/index.php/Distances_Computation)

- Image Based Metrics

One can use the [eval.sh](http://eval.sh) or eval_in_parallel.sh to evaluate, do not forget to modify the dataset name, and do add —eval in training procedure

```bash
Model_Folder=(
# modify the colmap
    /home/xiongbutian/workspace/SAM-GS/output_alpha_constrain_eval/SZTU_COLMAP
)
```
