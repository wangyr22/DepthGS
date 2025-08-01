import sys
sys.path.insert(0, "./SplaTAM")

import torch
from torchvision.utils import save_image
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
# from utils.recon_helpers import setup_camera, pose_to_transformation_matrix
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    _transform_to_frame, l1_loss_v1, matrix_to_quaternion
)
from scipy.spatial.transform import Rotation as R

import cv2
import numpy as np

import pdb

def gaussian_images(pose_t_2, pose_t_1, gaussian_t_1, image_t, intrinsics, cam, tr_ratio=1.0, nums_new=6, theta=30.):
    pose_t_1 = pose_t_1.detach().clone()
    pose_t_2 = pose_t_2.detach().clone()
    pose_t_1 = pose_t_1[[0, 1, 2, 4, 5, 6, 3]]
    pose_t_2 = pose_t_2[[0, 1, 2, 4, 5, 6, 3]]
    _pose_t_1 = pose_t_1
    # gaussians = {key: value.to('cuda').squeeze(0) for key, value in gaussian_t_1.items()}
    # images = image_t.to("cuda:0")
    # intrinsics = intrinsics.to('cuda')

    # pose
    matrix_t_minus_2 = pose_to_matrix(*pose_t_2.detach().cpu())
    matrix_t_minus_1 = pose_to_matrix(*pose_t_1.detach().cpu())

    # pose t
    new_translation2 = -1 * (torch.linalg.inv(matrix_t_minus_1) @ matrix_t_minus_2)[:3, 3]
    new_translation2 = new_translation2.to("cuda:0")
    # delta_translation = dpose_t_1[:3] - dpose_t_2[:3]
    # new_translation2 = dpose_t_1[:3] + delta_translation
    # pose_t = torch.cat((new_translation2, dpose_t_1[3:]))

    # 6 pose t new
    # these are translation.
    vectors_t = generate_cone_vectors(new_translation2 * tr_ratio, theta, num_vectors=nums_new)
    # this is just identity rotation.
    rotation_vector = torch.tensor([[0, 0, 0, 1.]], device="cuda:0").expand(vectors_t.size(0), -1)
    combined_vectors = torch.cat((vectors_t, rotation_vector), dim=1)

    combined_delta_vectors = []
    combined_vectors = []
    for vt in vectors_t:
        delta_motion = torch.eye(4)
        delta_motion[:3, 3] = vt
        motion = matrix_t_minus_1 @ delta_motion
        combined_delta_vectors.append( 
            torch.tensor(matrix_to_pose(delta_motion), device="cuda:0") )
        combined_vectors.append( 
            torch.tensor(matrix_to_pose(motion), device="cuda:0") )
    combined_delta_vectors = torch.stack(combined_delta_vectors)
    combined_vectors = torch.stack(combined_vectors)

    # 6 gaussian images
    gaussian_images, gaussian_depths = process_gaussian_images(combined_vectors, image_t, intrinsics, gaussian_t_1, pose_t_1, cam)
    gaussian_depths = torch.stack(gaussian_depths)
    gaussian_images = torch.stack(gaussian_images)
    # gaussian_images = gaussian_images[:, [2, 1, 0], :, :]
    gaussian_images = (gaussian_images * 255).to(torch.uint8)
    # image_t = image_t / 255.0

    return gaussian_images, gaussian_depths, combined_vectors


def process_gaussian_images(Gs, images, intrinsics, gaussians, pose_t_1, cam):
    # Setup camera
    w2c = torch.linalg.inv(pose_to_transformation_matrix(pose_t_1)) # pose_t_1 -> relative_pose_ab
    # cam = setup_camera(images.shape[-1], images.shape[-2], intrinsics, w2c)

    curr_pose_data = Gs.cuda()
    gaussians_images = []
    gaussians_depths = []
    
    for i in range(len(Gs)): 
        # Transform gaussians to the current frame
        transformed_gaussians = _transform_to_frame(gaussians, curr_pose_data[i].unsqueeze(0), 
                                                gaussians_grad=False, camera_grad=False)
        
        # Render transformed gaussians
        rendervar = transformed_params2rendervar(gaussians, transformed_gaussians)
        im, radius, _ = Renderer(raster_settings=cam)(**rendervar)
        gaussians_images.append(im)

        depth_sil_rendervar = transformed_params2depthplussilhouette(gaussians, w2c.cuda(), transformed_gaussians)
        depth_sil, _, _, = Renderer(raster_settings=cam)(**depth_sil_rendervar)
        depth = depth_sil[0, :, :].unsqueeze(0)
        gaussians_depths.append(depth)

        # save_depth_visualization(depth[0].detach().cpu().numpy(), f'output_depth_t{i}.png')
        # save_image(im, f'output_image_t{i}.png')

    return gaussians_images, gaussians_depths

def generate_cone_vectors(V0, angle_x, num_vectors=6):
    V_unnorm = V0
    V0 = V0 / V0.norm()
    
    angle_x_rad = torch.tensor(angle_x * (torch.pi / 180.0))
    
    cos_angle = torch.cos(angle_x_rad)
    sin_angle = torch.sin(angle_x_rad)

    vectors = []
    for i in range(num_vectors):
        theta = torch.tensor(2 * torch.pi * i / num_vectors).to(V0.device)
        
        new_vector = torch.tensor([
            cos_angle * V0[0] + sin_angle * torch.cos(theta),
            cos_angle * V0[1] + sin_angle * torch.sin(theta),
            sin_angle * V0[2]
        ]).to(V0.device)
        
        vectors.append((new_vector / new_vector.norm())*V_unnorm.norm())

    return torch.stack(vectors)

def pose_to_matrix(tx, ty, tz, qx, qy, qz, qw):
    """Convert pose to 4x4 transformation matrix using PyTorch."""
    rotation_matrix = torch.tensor(R.from_quat([qx, qy, qz, qw]).as_matrix(), dtype=torch.float32)
    transformation_matrix = torch.eye(4, dtype=torch.float32)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = torch.tensor([tx, ty, tz], dtype=torch.float32)
    return transformation_matrix

def matrix_to_pose(matrix):
    """Convert 4x4 transformation matrix back to pose format [tx, ty, tz, qx, qy, qz, qw] using PyTorch."""
    tx, ty, tz = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3].numpy()  # Convert to NumPy for quaternion conversion
    rotation = R.from_matrix(rotation_matrix)
    qx, qy, qz, qw = rotation.as_quat()
    return [tx.item(), ty.item(), tz.item(), qx, qy, qz, qw]


def save_depth_visualization(depth_map, output_path, colormap=cv2.COLORMAP_JET):
    normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_mapped_depth = cv2.applyColorMap(normalized_depth, colormap)
    cv2.imwrite(output_path, color_mapped_depth)


def setup_camera(w, h, k, w2c, near=0.01, far=100):
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    # print(k)
    # fx, fy, cx, cy = k[0], k[1], k[2], k[3]
    # w2c = torch.tensor(w2c).cuda().float()
    w2c = w2c.detach().clone().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    # w2c = w2c.unsqueeze(0).transpose(1, 2)
    w2c = w2c.unsqueeze(0).transpose(1, 2).cuda()
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c.float(),
        projmatrix=full_proj.float(),
        sh_degree=0,
        campos=cam_center,
        prefiltered=False
    )
    # cam = Camera(
    #     image_height=h,
    #     image_width=w,
    #     tanfovx=w / (2 * fx),
    #     tanfovy=h / (2 * fy),
    #     bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
    #     scale_modifier=1.0,
    #     viewmatrix=torch.tensor([[[ 1.0000e+00,  3.7253e-09,  5.5879e-09,  0.0000e+00],
    #      [-7.4506e-09,  1.0000e+00, -4.1633e-17,  0.0000e+00],
    #      [ 0.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00],
    #      [-1.1921e-07,  1.1921e-07, -5.9605e-08,  1.0000e+00]]],
    #    device='cuda:0'),
    #     projmatrix=torch.tensor([[[ 1.6278e+00,  8.3128e-09,  5.5885e-09,  5.5879e-09],
    #      [-1.2128e-08,  2.1708e+00, -4.1638e-17, -4.1633e-17],
    #      [ 1.5938e-02,  4.0417e-02,  1.0001e+00,  1.0000e+00],
    #      [-1.9500e-07,  2.5637e-07, -1.0001e-02, -5.9605e-08]]],
    #    device='cuda:0'),
    #     sh_degree=0,
    #     campos=cam_center,
    #     prefiltered=False
    # )
    return cam

def quaternion_to_rotation_matrix(px, py, pz, pw):
    R = torch.tensor([
        [1 - 2 * (py**2 + pz**2), 2 * (px * py - pz * pw), 2 * (px * pz + py * pw)],
        [2 * (px * py + pz * pw), 1 - 2 * (px**2 + pz**2), 2 * (py * pz - px * pw)],
        [2 * (px * pz - py * pw), 2 * (py * pz + px * pw), 1 - 2 * (px**2 + py**2)]
    ])
    return R

def pose_to_transformation_matrix(pose):
    x, y, z, px, py, pz, pw = pose
    R = quaternion_to_rotation_matrix(px, py, pz, pw)
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = torch.tensor([x, y, z])
    return T

