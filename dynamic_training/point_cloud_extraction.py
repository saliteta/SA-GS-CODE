import torch
import numpy as np
import open3d as o3d
from tqdm import trange
import torch.distributions as dist

from argparse import ArgumentParser
from scene.gaussian_model import GaussianModel

from utils.general_utils import build_scaling_rotation

'''
    This script is written by Butian Xiong butianxiong@link.cuhk.edu.cn
    This are the basic idea. We use alpha and Gaussian denstiy combine together as a known distribution, and we sample multipple points 
    from the distribution.

    We first build up a multinominal distribution according to alpha value. Note that we need to normalize the alpha
    alpha = alpha / alpha.sum()

    We sample N points from this multinominal distribution. The value corresponds to which Gaussian we need to sample from
    Then we sample according to Gaussian distribution. 

    Notice that directly sample all points in parallel will lead to CUDA OUT OF MEMORY
    Therefore, we utilize a batch strategy
'''

def add_jitter(cov_matrices, scale_factor=1e-6):
    # Calculate the mean or minimum of the diagonal elements across each matrix
    diagonal_values = torch.diagonal(cov_matrices, dim1=-2, dim2=-1)
    mean_diag = torch.mean(diagonal_values, dim=-1)
    jitter_values = mean_diag * scale_factor
    
    # Create a jitter matrix for each covariance matrix in the batch
    jitter_matrices = torch.eye(cov_matrices.size(-1), device=cov_matrices.device).unsqueeze(0) * jitter_values.unsqueeze(-1).unsqueeze(-1)
    adjusted_matrices = cov_matrices + jitter_matrices
    return adjusted_matrices

def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def multinominal_sample(gs: GaussianModel, num_sample_points):
    alpha = gs.get_opacity

    alpha = alpha.squeeze()

    sampled_gaussian_index = torch.multinomial(alpha, num_sample_points, replacement=True)

    return sampled_gaussian_index

def sample_from_gaussian_splats(gs:GaussianModel, sampled_gaussian_index: torch.Tensor) -> torch.Tensor:
    means = gs.get_xyz[sampled_gaussian_index]
    covairance_gaussian = build_covariance_from_scaling_rotation(gs.get_scaling, 1.0, gs.get_rotation)
    covairance = covairance_gaussian[sampled_gaussian_index]
    
    covairance = add_jitter(covairance)

    mvn = dist.MultivariateNormal(means, covariance_matrix=covairance)


    samples = mvn.sample()

    return samples



def main():
    parser = ArgumentParser(description="Sampling parameters for point cloud generation")
    parser.add_argument('--gs_model', type=str, default="../output_alpha_constrain/CUHK_LOWER_CAMPUS_COLMAP/point_cloud/iteration_30000/point_cloud.ply")
    parser.add_argument('--store_location', type=str, default="../extract_point_cloud/cuhk_lower.ply")
    parser.add_argument("--points_number", type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--color', action='store_true', default=False)

    args = parser.parse_args()
    if args.color:
        raise NotImplementedError("Color sampling not implemented.")

    # Load Gaussian Model
    gs = GaussianModel(3)  # SH level usually is set to 3
    gs.load_ply(args.gs_model)

    # Prepare for batch processing
    num_batches = (args.points_number + args.batch_size - 1) // args.batch_size
    all_points = np.zeros((args.points_number, 3), dtype=np.float32)
    with torch.no_grad():
        for i in trange(num_batches):
            current_batch_size = min(args.batch_size, args.points_number - i * args.batch_size)

            # Sample indices from the multinomial distribution
            sampled_gaussian_index = multinominal_sample(gs, current_batch_size)

            # Sample points from the selected Gaussian splats
            batch_points = sample_from_gaussian_splats(gs, sampled_gaussian_index)

            # Convert tensor to numpy and store in the array
            start_idx = i * args.batch_size
            end_idx = start_idx + current_batch_size
            all_points[start_idx:end_idx, :] = batch_points.cpu().numpy()

    # Create a point cloud object and store as PLY
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    o3d.io.write_point_cloud(args.store_location, pcd)
    print(f"Point cloud saved to {args.store_location}")

main()
    

    




