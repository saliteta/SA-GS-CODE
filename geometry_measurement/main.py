from dataloader import maskedDataset
from complexity import complexity
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd


def argparser():

    parser = argparse.ArgumentParser(description="Visualize and FFT to original image to specturm magnitude")
    parser.add_argument("--prompts_file", type=str, required=True, help="the path to original prompts file")
    parser.add_argument("--output_file", type=str, required=True, help="the path to store the output csv file")
    parser.add_argument("--image_path", type=str, required=True, help="the path to image folder")
    parser.add_argument("--mask_path", type=str, required=True, help="the path to mask folder")
    return parser

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def main():

    parser = argparser()

    args = parser.parse_args()

    print(f"image path: {args.image_path}")
    print(f"mask path: {args.mask_path}")

    img_metrix = complexity()

    masked_dataset = maskedDataset(image_dir=args.image_path, mask_dir=args.mask_path)


    edge = []
    pixel_number_list = []
    count = 1
    for images, name in tqdm(masked_dataset):
        edge_size, pixel_number = img_metrix.edge_analysis(images)
        #print(f'overall ({count}/{len(masked_dataset)}), we have :{edge_size} as edge_size, {pixel_number} as pixel number, {name} as name')
        edge.append(edge_size)
        pixel_number_list.append(pixel_number)
        count += 1

    edge = np.array(edge)
    pixel_number = np.array(pixel_number_list)

    edge_sum = edge.sum(axis=0)
    pixel_number_sum = pixel_number.sum(axis=0)
    
    with open(args.prompts_file, 'r') as file:
        prompts = [line.strip() for line in file]
    
        
    print(f"edge complexity including others at the first element: {edge_sum}")
    print(f"pixel_number_sum complexity including others at the first element: {pixel_number_sum}")
    print(f"final result including others at the first element is {edge_sum/pixel_number_sum}")
    
    # Check to avoid division by zero and ensure lists are of equal length
    if len(edge_sum) == (len(prompts)+1) and all(pixel_number_sum):
        data = {prompts[i]: edge_sum[i+1] / pixel_number_sum[i+1] for i in range(len(prompts))}
        df = pd.DataFrame([data])  # Convert dictionary into a DataFrame

        # Save to CSV
        df.to_csv(args.output_file, index=False)
        print(f"Data saved to f{args.output_file}")
    else:
        print("Error: Lists are not the same length or division by zero might occur.")

if __name__ == "__main__":
    main()
    
