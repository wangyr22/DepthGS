import sys
sys.path.append("./DROID-SLAM/droid_slam")

import argparse

import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation as R

from droid import Droid
import geom.projective_ops as pops
from modules.corr import CorrBlock

_buffer_size = 8

# pseudo args
_args = argparse.Namespace(
    imagedir="",
    calib="",
    t0="",
    stride=1,
    weights="./DROID-SLAM/droid.pth",
    # we only need 2 frames.
    buffer=_buffer_size,
    image_size=[480, 640],
    disable_vis=True,
    beta=0.3,
    filter_thresh=2.4,
    warmup=8,
    keyframe_thresh=4.0,
    frontend_thresh=16.0,
    frontend_window=25,
    frontend_radius=2,
    frontend_nms=1,
    backend_thresh=22.0,
    backend_radius=2,
    backend_nms=3,
    upsample=False,
    reconstruction_path="",
    stereo=False
)

STDV = torch.as_tensor([0.229, 0.224, 0.225], device="cuda:0")[:, None, None]
MEAN = torch.as_tensor([0.485, 0.456, 0.406], device="cuda:0")[:, None, None]

def pose_matrix_from_quaternion(pvec):
    """ convert 4x4 pose matrix to (t, q) """
    pose = np.eye(4)
    pose[:3, :3] = R.from_quat(pvec[3:]).as_matrix()
    pose[:3, 3] = pvec[:3]
    return pose

class PoseEstimator:
    def __init__(self, intrinsics, image_size, orig_size, buffer_size=None):
        _args.image_size = image_size
        if buffer_size is not None:
            _args.buffer = buffer_size
        self.droid = Droid(_args)
        self._data_all_ref = []

        self.ht = image_size[0]
        self.wd = image_size[1]
        self.ht_orig = orig_size[0]
        self.wd_orig = orig_size[1]
        # [fx, fy, cx, cy]
        x_ratio = self.wd / self.wd_orig
        y_ratio = self.ht / self.ht_orig
        self.intrinsics = intrinsics * torch.tensor([x_ratio, y_ratio, x_ratio, y_ratio])
        if self.ht == self.ht_orig and self.wd == self.wd_orig:
            self.need_resize = False
        else:
            self.need_resize = True

        self.eps = 1e-6
        self.dmax = 20

    def _reset_all(self):
        device = torch.device("cuda:0")
        g = self.droid.frontend.graph
        ht, wd = g.ht, g.wd
        self.droid.frontend.graph.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.droid.frontend.graph.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.droid.frontend.graph.age = torch.as_tensor([], dtype=torch.long, device=device)
        self.droid.frontend.graph.corr = None
        self.droid.frontend.graph.net = None
        self.droid.frontend.graph.inp = None
        self.droid.frontend.graph.target = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.droid.frontend.graph.weight = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

    @torch.no_grad()
    def initialize_first_frame(self, img0, depth0, first_frame_w2c):
        self.c2w_accumulated = np.linalg.inv(first_frame_w2c.detach().cpu().numpy())
        # self.h, self.w = img0.shape[1:]
        if self.need_resize:
            img0 = F.interpolate(img0[None], [self.ht, self.wd], mode="bilinear")[0]
            depth0 = F.interpolate(depth0[None], [self.ht, self.wd], mode="bilinear")[0]

        img = img0
        inputs = img[None, None].sub(MEAN).div(STDV)
        depth = depth0[0]
        disp = torch.clip(
            1 / (depth + self.eps)[3::8, 3::8], min=0, max=self.dmax)
        gmap = self.droid.filterx._MotionFilter__feature_encoder(inputs)
        net, inp = self.droid.filterx._MotionFilter__context_encoder(inputs[:, [0]])

        self._data_prev = {
            "img": img0,
            "inputs": inputs,
            "depth": depth,
            "depth_mask": (depth > 0),
            "disp": disp,
            "gmap": gmap,
            "net": net,
            "inp": inp,
            "pvec": torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float32, device="cuda:0")
        }

        self._data_cur = None

    @torch.no_grad()
    def build_reference(self, imgs: list[torch.Tensor], depths: list[torch.Tensor], pvecs: list[torch.Tensor]):
        _ref_nums = len(imgs)
        imgs = [
            F.interpolate(img[None], [self.ht, self.wd], mode="bilinear")[0] for img in imgs
        ]
        inputs = [
            img[None, None].sub(MEAN).div(STDV) for img in imgs
        ]
        depths = torch.stack(depths)
        depths = F.interpolate(depths, [self.ht, self.wd], mode="bilinear")
        # disps = torch.clip(
        #     1 / F.interpolate(depths + self.eps, size=[self.ht // 8, self.wd // 8]), min=0, max=self.dmax)
        # print(f"ref: depths.shape: {depths.shape}")
        disps = torch.clip(
            1 / (depths + self.eps)[..., 3::8, 3::8], min=0, max=self.dmax)
        depths = depths[:, 0, ...].unbind()
        disps = disps[:, 0, ...].unbind()

        self._data_all_ref = []
        for i in range(_ref_nums):
            gmap = self.droid.filterx._MotionFilter__feature_encoder(inputs[i])
            net, inp = self.droid.filterx._MotionFilter__context_encoder(inputs[i][:,[0]])
            self._data_all_ref.append(
                {
                    "img": imgs[i],
                    "inputs": inputs[i],
                    "depth": depths[i],
                    "depth_mask": (depths[i] > 0),
                    "disp": disps[i],
                    "gmap": gmap,
                    "net": net,
                    "inp": inp,
                    "pvec": pvecs[i]
                })

        self._ref_pvecs = torch.stack(pvecs)
        
    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def filter(self, img_cur, depth_cur, thresh=2.5):
        if self.need_resize:
            img_cur = F.interpolate(img_cur[None], [self.ht, self.wd], mode="nearest")[0]
            depth_cur = F.interpolate(depth_cur[None], [self.ht, self.wd], mode="nearest")[0]

        img = img_cur
        inputs = img[None, None].sub(MEAN).div(STDV)
        depth = depth_cur[0]
        disp = torch.clip(
            1 / (depth + self.eps)[3::8, 3::8], min=0, max=self.dmax)
        gmap = self.droid.filterx._MotionFilter__feature_encoder(inputs)

        coords0 = pops.coords_grid(self.ht // 8, self.wd // 8, device="cuda:0")[None,None]
        corr = CorrBlock(self._data_prev["gmap"][None,[0]], gmap[None,[0]])(coords0)
        _, delta, weight = self.droid.net.update(
            self._data_prev["net"][None], self._data_prev["inp"][None], corr
        )

        if delta.norm(dim=-1).mean().item() <= thresh:
            return False
        
        net, inp = self.droid.filterx._MotionFilter__context_encoder(inputs[:,[0]])

        self._data_cur = {
            "img": img,
            "inputs": inputs,
            "depth": depth,
            "depth_mask": (depth > 0),
            "disp": disp,
            "gmap": gmap,
            "net": net,
            "inp": inp,
            "pvec": torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float32, device="cuda:0")
        }
        return True

    @torch.no_grad()
    def track(self, iters=12, gt_c2w=None):
        self._reset_all()

        for i, _data in enumerate([self._data_prev, *self._data_all_ref, self._data_cur]):
            # print(f"{i} depth_mask: {_data['depth_mask'].shape}")
            self.droid.video._DepthVideo__item_setter(
                i,
                [i, _data['img'][0],
                _data["pvec"],
                _data['disp'],
                _data['depth'],
                self.intrinsics / 8.0,
                _data['gmap'],
                _data['net'],
                _data['inp'],
                _data['depth_mask']])
        
        _ref_nums = len(self._data_all_ref)
        ref_total_nums = _ref_nums + 1
        ii = []
        jj = []
        for k in range(ref_total_nums):
            ii.extend([k, ref_total_nums])
            jj.extend([ref_total_nums, k])
        self.droid.frontend.graph.add_factors(ii, jj)
    
        for itr in range(iters):
            self.droid.frontend.graph.update(motion_only=True)
            # print(f"poses #{itr}: {self.droid.video.poses}")

            # reset pose of all reference frames
            if _ref_nums > 0:
                self.droid.video.poses[1: _ref_nums + 1] = self._ref_pvecs

            if gt_c2w is not None:
                est_now = self.droid.video.poses[_ref_nums + 1].detach().cpu().numpy()
                est_now = pose_matrix_from_quaternion(est_now)
                est_now = np.linalg.inv(est_now)
                trl2_err = trans_l2_err(est_now[:3, -1], gt_c2w[:3, -1])
                rotd_err = rot_degree_err(est_now[:3, :3], gt_c2w[:3, :3])
                print(f"iter {itr}: {trl2_err}, {rotd_err}")

        # update memory
        self._data_prev = self._data_cur
        self._data_cur = None
        self._data_all_ref = []
        self._ref_pvecs = None

        # this is quaternion w2c.
        pvec = self.droid.video.poses[_ref_nums + 1].detach().cpu().numpy()
        # print(f"poses: {self.droid.video.poses}")
        cam_to_prevcam = np.linalg.inv(pose_matrix_from_quaternion(pvec))

        c2w_accumulated = self.c2w_accumulated @ cam_to_prevcam
        self.c2w_accumulated = c2w_accumulated

        return c2w_accumulated, cam_to_prevcam

def pose_matrix_from_quaternion(pvec):
    """ convert 4x4 pose matrix to (t, q) """
    pose = np.eye(4)
    pose[:3, :3] = R.from_quat(pvec[3:]).as_matrix()
    pose[:3, 3] = pvec[:3]
    return pose

def trans_l2_err(trans, gt):
    return np.sqrt( np.square(trans - gt).sum() )

def rot_degree_err(rot, gt):
    # pass in as matrices
    rot_err = np.linalg.inv(gt) @ rot
    return np.arccos(
        min(1, max(-1, (np.trace(rot_err) - 1) / 2))) * 180 / np.pi
