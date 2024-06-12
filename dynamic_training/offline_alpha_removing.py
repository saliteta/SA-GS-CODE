import os
import math
import torch
import argparse
import numpy as np
import open3d as o3d
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readColmapSceneInfo
from utils.camera_utils import cameraList_from_camInfos
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


### This code is written by Butian Xiong ####
# - We calculate how many -1, 0, 1, 2 we have (in numpy file, we can do this in pixel level)
# - We calculate how many -1, 0, 1, 2 we have in Gaussian Spalts level (We can do this in rasterization)
# - We calculate the alpha value distribution for different categories (Statistic for 20 random selected picture, and overall)
# - We display the result, we remove according to the alpha value and display it 
# - If we have a threshould, we set it as the lower bound, and we set a stablize differentiable equation to converge (PID) controller

'''
We need to generate good visualization result
- Visualization in Statistic way
- Visualization in ply according to alpha value, if its alpha value is large, it becomes red, otherwise, blue
- Alpha value statisic also
'''


explaination = "This code is used for data analysis when we do the alpha filtering in training. \n\
                to be specific: we calculate semantic mask and alpha value, and do the statistic "

def render(viewpoint_camera, pc : GaussianModel, bg_color : torch.Tensor = torch.Tensor([0, 0, 0]).cuda(), scaling_modifier = 1.0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    colors_precomp = None
    shs = pc.get_features

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, Means2D = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "Means2D": Means2D}

def visualize_ply(confidence_score: np.ndarray, xyz: torch.Tensor, result_location: str):
    # Ensure the result_location exists
    os.makedirs(result_location, exist_ok=True)
    
    # Convert tensor to numpy if it's not already in that format
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.cpu().numpy()
    
    # Calculate the mean and standard deviation
    mean = np.mean(confidence_score)
    std = np.std(confidence_score)
    
    # Normalize confidence scores using 3-sigma rule
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    # Clip scores and then scale to 0-1
    clipped_scores = np.clip(confidence_score, lower_bound, upper_bound)
    normalized_scores = (clipped_scores - lower_bound) / (upper_bound - lower_bound)

    # Use a Matplotlib colormap that goes from blue to red
    cmap = plt.get_cmap('coolwarm')
    colors = cmap(normalized_scores)[:, :3]  # Ignore the alpha channel

    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Optionally visualize the point cloud (comment out if running on a headless server)
    # o3d.visualization.draw_geometries([pcd])

    # Save to PLY file
    ply_file_path = os.path.join(result_location, 'heat_visualization.ply')
    o3d.io.write_point_cloud(ply_file_path, pcd)
    print(f"Point cloud saved to {ply_file_path}")

def gs_alpha_per_class(gs:GaussianModel, category, categories: torch.Tensor, path_to_save:str, catagoric_name:str = 'ground'):
    # We mark which Gaussian is in which class, and we statistic them according to alpha value one class a time (inconsistancy)
    opacity = gs.get_opacity[torch.where(categories==category)]
    # get means and print
    opacity = opacity.cpu().numpy()

    hist_generation(opacity, path_to_save, plot_name=catagoric_name)

def hist_generation(data: np.ndarray, result_location: str, plot_name):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(data, bins=30, color='skyblue', kde=True) 
    plt.title(plot_name)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(result_location, f'{plot_name}.png'))

def gs_per_class(semantic_mask: torch.Tensor, Means2D: torch.Tensor, categories_number: int):
    '''
    Retrieve indices from Means2D where the means correspond to a specific category in the semantic mask.

    Arguments:
    - semantic_mask: Tensor of semantic masks.
    - Means2D: Tensor containing means, assumed to be of shape [n, 2] where each row is [x, y].
    - categories_number: Integer representing the desired category number.

    Returns:
    - indices: Tensor of indices in Means2D that correspond to the specified category number.
    '''

    # Ensure all tensors are on the same device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    semantic_mask = semantic_mask.to(device)
    Means2D = Means2D.to(device)

    # Determine valid indices where Means2D is not [0.0, 0.0] and is within the bounds of the semantic mask
    valid_indices = ~(torch.all(Means2D == torch.tensor([0.0, 0.0], device=device), dim=1)) & \
                    (Means2D[:, 0] >= -semantic_mask.shape[0] // 2) & \
                    (Means2D[:, 0] < semantic_mask.shape[0] // 2) & \
                    (Means2D[:, 1] >= -semantic_mask.shape[1] // 2) & \
                    (Means2D[:, 1] < semantic_mask.shape[1] // 2)

    valid_means = Means2D[valid_indices]

    # Convert mean coordinates to integer indices
    x = valid_means[:, 0].long() + semantic_mask.shape[0] // 2
    y = valid_means[:, 1].long() + semantic_mask.shape[1] // 2

    # Check if indices are within bounds
    assert x.min() >= 0 and x.max() < semantic_mask.shape[0], "x indices are out of bounds"
    assert y.min() >= 0 and y.max() < semantic_mask.shape[1], "y indices are out of bounds"

    # Extract the semantic mask values for the valid means
    mask = semantic_mask[x, y]

    # Retrieve indices corresponding to the specific category number
    category_mask_indices = torch.where(mask == categories_number)[0]

    # Map these back to original indices in Means2D
    indices = torch.nonzero(valid_indices, as_tuple=True)[0][category_mask_indices]

    return indices.long()



def main():
    
    parser = argparse.ArgumentParser(description=explaination)
    
    parser.add_argument('--colmap_folder', type=str)
    parser.add_argument('--gaussian_model', type=str)
    parser.add_argument('--result_folder_location', type=str)
    parser.add_argument('--SH_degree', type=int, default=3) # sh degree is 3
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--images', type=str, default='images')
    parser.add_argument('--data_device', type=str, default='cpu')
    parser.add_argument('--categorical_number', type=int, default=3)

    args = parser.parse_args()

    # load model
    gs = GaussianModel(args.SH_degree) 
    gs.load_ply(args.gaussian_model)

    os.makedirs(args.result_folder_location, exist_ok=True)

    class_name = ['ground', 'building', 'vegetation']

    # load camera info
    scene_info = readColmapSceneInfo(path=args.colmap_folder, images=args.images, eval=None)
    camera_list = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale=1, args=args)
    for i in range(3):
        categories = torch.zeros(len(gs.get_xyz), dtype=torch.int32).cuda()
        with torch.no_grad():
            # now is for Gaussian Render
            for view in tqdm(camera_list):
                means2D = render(view, gs)['Means2D']
                valid_index = gs_per_class(semantic_mask=view.gt_semantic_mask, Means2D = means2D, categories_number=i)
                categories[valid_index] = i+1
                print(categories.sum())
            gs_alpha_per_class(gs=gs, 
                               category=i+1,
                               categories=categories, 
                               path_to_save=os.path.join(args.result_folder_location), 
                               catagoric_name=class_name[i])





if __name__ == "__main__":
    main()








