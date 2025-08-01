import os
from os.path import join as pjoin
from typing import Optional

import numpy as np
import cv2
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset

def create_filepath_index_mapping(frames):
    return {frame["file_path"] + '.png': index for index, frame in enumerate(frames)}

class CustomDataset(GradSLAMDataset):
    def __init__(
        self,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 1440,
        desired_width: Optional[int] = 1920,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        config_dict = {}
        config_dict["dataset_name"] = "custom"

        # Load RGB & Depth filepaths
        self.image_names = natsorted(os.listdir(f"{self.input_folder}/color"))
        self.image_names = [f'color/{image_name}' for image_name in self.image_names]

        # Init Intrinsics
        fx, fy, cx, cy = np.loadtxt(pjoin(self.input_folder, 'intrinsics.txt')).tolist()
        ht, wd, _ = cv2.imread( pjoin(self.input_folder, self.image_names[0]) ).shape

        config_dict["camera_params"] = {}
        config_dict["camera_params"]["png_depth_scale"] = 1.
        config_dict["camera_params"]["image_height"] = ht
        config_dict["camera_params"]["image_width"] = wd
        config_dict["camera_params"]["fx"] = fx
        config_dict["camera_params"]["fy"] = fy
        config_dict["camera_params"]["cx"] = cx
        config_dict["camera_params"]["cy"] = cy

        print(f"camera params: {config_dict['camera_params']}")

        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        ) 
    
    def get_filepaths(self):
        print('NOTE: Using identity matrices as pose placeholders. In this case, ATE RMSE metric is not meaningful.')

        base_path = f"{self.input_folder}"
        color_paths = []
        depth_paths = []
        self.tmp_poses = []
        for image_name in self.image_names:
            # Get path of image and depth
            color_path = f"{base_path}/{image_name}"
            color_paths.append(color_path)
            if self.use_unidepth:
                depth_path = f"{base_path}/{image_name.replace('color', 'depth').replace('png', 'npy')}"
            else:
                depth_path = f"{base_path}/{image_name.replace('color', 'depth')}"
            depth_paths.append(depth_path)
            # don't have pose. use some identitiy matrices as placeholder
            self.tmp_poses.append(torch.eye(4))
        embedding_paths = None

        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        return self.tmp_poses

    def read_embedding_from_file(self, embedding_file_path):
        raise NotImplementedError
