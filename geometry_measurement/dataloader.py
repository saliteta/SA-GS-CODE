import os
import numpy as np
import cv2
from torch.utils.data import Dataset

class maskedDataset(Dataset):
    def __init__(self, mask_dir, image_dir, semantic_mask_number = 3, transform=None):
        self.mask_dir = mask_dir
        self.image_dir = image_dir
        self.semantic_mask_number = semantic_mask_number
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npz')])
        self.image_files = sorted([f for f in os.listdir(image_dir) if (f.endswith('.jpg') or f.endswith('.JPG'))])
        self.transform = transform

    def __len__(self):
        return min(len(self.mask_files), len(self.image_files))

    def __getitem__(self, idx):
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        
        
        mask = np.load(mask_path)['arr_0']
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if image.shape[2] == 4:  # Check if image has an alpha channel
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Get unique mask values and filter out -1
        mask_order_number = np.arange(-1, self.semantic_mask_number)


        # Prepare output tensor
        result_images = []

        for val in mask_order_number:
            mask_channel = (mask == val).astype(np.uint8)
            channel_image = np.zeros_like(image)
            if self.transform:
                image = self.transform(image)
            for i in range(3):  # Assuming image has 3 channels
                channel_image[:, :, i] = image[:, :, i] * mask_channel

            result_images.append(channel_image)


        return np.array(result_images), self.mask_files[idx]

# Loader test has passed

