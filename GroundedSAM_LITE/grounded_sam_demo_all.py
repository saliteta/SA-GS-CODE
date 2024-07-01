import argparse
import os
import sys

import numpy as np
import json
import torch
from PIL import Image

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.multiprocessing as multiprocessing

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return np.array(image_pil), image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)


    filt_mask = logits.max(dim=1)[0] > box_threshold
    boxes_filt = boxes[filt_mask]  # num_filt, 4


    return boxes_filt

def debuging_save_jpg(output_dir, mask_list, basename, text_prompt):
    value = 0  # 0 for background
    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir,'img', basename[0] + '_' + text_prompt + '.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
    
def per_caption_data(mask_list, value):
    mask_img = torch.zeros(mask_list.shape[-2:])
    for mask in mask_list:
        mask_img[mask.cpu().numpy()[0] == True] = value
    return mask_img

def parser():
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--prompts_file", type=str, required=True, help="path to the text file that stores the prompt file")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h")
    parser.add_argument("--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file")
    parser.add_argument("--image_dir", type=str, required=True, help="path to image file")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")
    parser.add_argument("--debugging", type=str, default="False", help="if to save the internal output to image folder")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--batch_num", type=str, default=16, help="The number of images manipulating at one time, default=64")
    args = parser.parse_args()
    return args

def sam(image_path, model, predictor, prompts, debugging, args):
    # cfg
    image_dir = args.image_dir
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    basename = image_path.split('.')
    # load image
    image_path = image_dir + image_path
    image, image_origin = load_image(image_path)
    size = image.shape
    predictor.set_image(image)
    H, W = size[0], size[1]
    masks_prompt_list = []
    masks_prompt_list.append(torch.zeros((H,W)))
    for idx, text_prompt in enumerate(prompts):
        # run grounding dino model
        boxes_filt = get_grounding_output(
            model, image_origin, text_prompt, box_threshold, text_threshold, device=device
        )
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H]).cuda()
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        boxes_filt = boxes_filt
        try: 
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
            masks, _, _ = predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(device),
                multimask_output = False,
            )
            if debugging == 'True':
                debuging_save_jpg(output_dir, masks, basename, text_prompt)
            masks_prompt_list.append(per_caption_data(masks, idx+1))
        except:
            continue

    final_mask = torch.max(torch.stack(masks_prompt_list),dim=0)[0]-1
    np.savez(os.path.join(output_dir,'npz',basename[0]), final_mask.cpu().numpy())
    
    
    

if __name__ == "__main__":

    args = parser()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    image_dir = args.image_dir
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
    batch_num = args.batch_num
    debugging = args.debugging
    prompts_file = args.prompts_file

    with open(prompts_file, 'r') as file:
        prompts = [line.strip() for line in file]
    print('your prompt is the following: ', prompts)
    
    # make dir
    print('Loading model')
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    print('Initialize Grounding Dino')
    # initialize SAM
    predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    print('Initialize SamPredictro')


    print('model initialization is accomplished')    
    print(f'The output file will be in folder: {output_dir}')    
    for image_path in tqdm(sorted(os.listdir(image_dir))):
        sam(image_path, model, predictor, prompts, debugging, args)