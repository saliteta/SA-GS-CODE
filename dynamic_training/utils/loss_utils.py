#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output:torch.Tensor, gt: torch.Tensor):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

## This code is implemented by Butian Xiong, the basic idea is to encourage the disk shape ####
## k1 should be larger than k2
# Basically is a weighted loss function that encourage some part of Gaussian to be some shape
def shape_constrain_loss(perplexity: torch.Tensor, semantic_mask: torch.Tensor, Means2D: torch.Tensor, scalings: torch.Tensor, k1=3, k2=1):
    '''
    Calculate a shape constraint loss based on the description provided.
    
    Arguments:
    - perplexity: Tensor containing perplexity values.
    - semantic_mask: Tensor of semantic masks.
    - Means2D: Tensor containing means, assumed to be of shape [n, 2] where each row is [x, y].
    - scalings: Tensor of scalings for Gaussian models.
    - k1, k2: Scaling factors for the loss components.
    '''

    # Ensure all tensors are on the same device, ideally 'cuda' for GPU acceleration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    perplexity = perplexity.to(device)
    semantic_mask = semantic_mask.to(device)
    Means2D = Means2D.to(device)
    scalings = scalings.to(device)

    # Filter valid Gaussians, assuming Means2D is [n, 2] and all are valid if not exactly [0,0] (0, 0) is the initial value
    # Calculate valid_indices considering [0,0] at the center of semantic_mask
    valid_indices = ~(torch.all(Means2D == torch.tensor([0.0, 0.0], device=device), dim=1)) & \
                    (Means2D[:, 0] >= -semantic_mask.shape[0]//2) & (Means2D[:, 0] < semantic_mask.shape[0]//2) & \
                    (Means2D[:, 1] >= -semantic_mask.shape[1]//2) & (Means2D[:, 1] < semantic_mask.shape[1]//2)

    valid_means = Means2D[valid_indices]

    # Convert mean coordinates to integer indices, ensuring they are within bounds
    x = valid_means[:, 0].long() + semantic_mask.shape[0] // 2
    y = valid_means[:, 1].long() + semantic_mask.shape[1] // 2


    # Get masks and filter out -1 (indicating no calculation needed)
    assert x.min() >= 0 and x.max() < semantic_mask.shape[0], "x indices are out of bounds"
    assert y.min() >= 0 and y.max() < semantic_mask.shape[1], "y indices are out of bounds"

    mask = semantic_mask[x, y]  # Note the order of indices [x,y]
    valid_mask_indices = mask != -1
    mask = mask[valid_mask_indices].long()

    if mask.numel() == 0:
        return torch.tensor(0.0, device=device)  # No valid data points to process

    # Get valid perplexities using mask
    p = perplexity[mask]

    # Filter valid scalings based on the mask
    valid_scalings = scalings[valid_indices][valid_mask_indices]
    a1 = valid_scalings[:, 0] / valid_scalings[:, 1]
    a2 = valid_scalings[:, 0] / valid_scalings[:, 2]

    # Compute the loss for each valid Gaussian
    loss_per_gaussian = torch.sigmoid(k1 / p - a1) + torch.sigmoid(k2 / p - a2)

    # Return the average loss or handle cases where there are no valid Gaussians
    return loss_per_gaussian.mean() if loss_per_gaussian.numel() > 0 else torch.tensor(0.0, device=device)
# Example usage:
# Assume you have initialized the tensors `perplexity`, `semantic_mask`, `Means2D`, and `scalings` appropriately.
# Call shape_constrain_loss(None, perplexity_tensor, semantic_mask_tensor, means2D_tensor, scalings_tensor)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

