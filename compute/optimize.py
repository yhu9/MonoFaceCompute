import os
import datetime
import json
import math
from pathlib import Path
from typing import List, Union

import mediapipe # fixes an error that occurs when importing some packages before mediapipe
import argparse
from omegaconf import OmegaConf
import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2

import inferno
from inferno.models.DecaFLAME import FLAME_mediapipe
from inferno.models.DECA import replace_asset_dirs
from inferno.models.Renderer import SRenderY
from inferno.utils.DecaUtils import tensor_vis_landmarks
import inferno.layers.losses.DecaLosses as lossfunc
import inferno.layers.losses.MediaPipeLandmarkLosses as lossfunc_mp

from utils import gaussian_kernel, apply_featurewise_conv1d

PER_FRAME_TRANSLATION = True

# Optimization
LR = 1e-2
LR_SHAPE = 1e-3
USE_FAN_LANDMARKS = True
USE_MP_LANDMARKS = True

def projection(points: Tensor, K: Tensor, w2c: Tensor) -> Tensor:
    """ Perspective camera projection. """

    rot = w2c[:, None, :3, :3]
    trans = w2c[:, None, :3, 3]
    points_cam = (points[..., None, :] * rot).sum(-1) + trans

    points_cam_projected = points_cam
    points_cam_projected[..., :2] /= points_cam[..., [2]]
    points_cam[..., 2] *= -1

    i = points_cam_projected[..., 0] * K[0] + K[2]
    j = points_cam_projected[..., 1] * K[1] + K[3]
    points2d = torch.stack([i, j, points_cam_projected[..., -1]], dim=-1)
    return points2d

def visualize(visdict):
    grids = {}
    for key in visdict:
        if visdict[key] is not None:
            grids[key] = torchvision.utils.make_grid(F.interpolate(visdict[key], [512, 512])).detach().cpu()
    grid = torch.cat(list(grids.values()), dim=1)
    grid_image = grid.permute(1,2,0).flip(-1) * 255 # CHW to HWC and RGB to BGR
    grid_image = grid_image.clamp(min=0, max=255).numpy().astype(np.uint8)
    return grid_image

def load_model(model_path, device, render_size=None):
    # Load a DECA config
    with open(model_path / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)
    conf = replace_asset_dirs(conf, model_path)
    cfg = conf["detail"].model
    # Use landmark embedding with eyes to optimize iris parameters
    cfg.flame_lmk_embedding_path = os.path.join("data", "landmark_embedding_with_eyes.npy")
    cfg.flame_mediapipe_lmk_embedding_path = os.path.join("data", "landmark_embedding_mediapipe.npz")
    # flame_cfg.n_shape = ...
    flame = FLAME_mediapipe(cfg)
    render = SRenderY(cfg.image_size if render_size is None else render_size, obj_filename=cfg.topology_path, uv_size=cfg.uv_size)
    return flame.to(device), render.to(device)


class Optimizer:
    def __init__(self, image_path: str, shape_from: Union[str, None], optimize_expr: bool, optimize_shape: bool, size: int,
                 smooth: bool, device='cuda:0'):
        self.image_path = image_path
        self.input_shape = None if shape_from is None else torch.tensor(json.load(open(shape_from, 'r'))["shape_params"]).float().to(device).unsqueeze(0)
        self.optimize_expr = optimize_expr
        self.optimize_shape = optimize_shape
        self.img_size = size
        self.smooth = smooth
        self.device = device

        # Load any version of DECA, this is just to have access to the rendering
        model_path = Path(inferno.__file__).parents[1] / "assets/EMOCA/models/DECA"
        self.flame, self.render = load_model(model_path, device, render_size=self.img_size)

    def optimize(self, shape, exp, landmarks_fan, landmarks_mp, pose, cam, names, visualize_images, visualize_sampling,
                 savefolder, intrinsics, save_name, use_iris, iterations):
        num_img = pose.shape[0]
        size = self.img_size
        device = self.device

        visualize_images = []
        for k, name in enumerate(names):
            if k % visualize_sampling == 0:
                image = cv2.imread(f"{self.image_path}/{name}.png").astype(np.float32) / 255.
                image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
                visualize_images.append(torch.from_numpy(image).to(device).unsqueeze(0))
        visualize_images = torch.cat(visualize_images, dim=0)

        # we need to project to [-1, 1] instead of [0, size], hence modifying the cam_intrinsics as below
        cam_intrinsics = torch.tensor(
            [-1 * intrinsics[0] / size * 2, intrinsics[1] / size * 2,
             intrinsics[2] / size * 2 - 1, intrinsics[3] / size * 2 - 1]).float().to(device)

        # Add neck rotation to the pose
        pose = torch.cat([pose[:,:3], torch.zeros_like(pose[:, :3]), pose[:,3:]], dim=1)
        # Add gaze direction to the pose
        if use_iris:
            pose = torch.cat([pose, torch.zeros_like(pose[:, :6])], dim=1)
        # Add per-frame translation
        if PER_FRAME_TRANSLATION:
            # Convert the cam predictions for the perspective camera model
            s,tx,ty = cam.unbind(-1)
            F = intrinsics[0] * 2 / size
            pose_translation = torch.stack([tx, ty, -F/s], dim=-1)
            pose = torch.cat((pose, pose_translation), dim=1)

        # The camera is fixed and the head translation/rotation is handled in FLAME
        ident = torch.eye(3, dtype=torch.float, device=device).unsqueeze(0).expand(num_img, -1, -1)
        w2c = torch.cat([ident, torch.zeros((num_img, 3, 1), dtype=torch.float, device=device)], dim=2)

        # pose now contains, in order: global rotation, neck, jaw, eye 1, eye 2, translation
        pose = Parameter(pose)
        exp = Parameter(exp)
        shape = Parameter(shape)

        # Create optimizer
        param_groups = []
        param_groups.append({"params": pose, "lr": LR})
        if self.optimize_expr: param_groups.append({"params": exp, "lr": LR})
        if self.optimize_shape: param_groups.append({"params": shape, "lr": LR_SHAPE})
        optimizer = torch.optim.Adam(param_groups)

        # Optimization
        for k in range(iterations+1):
            full_pose = pose
            if not use_iris:
                full_pose = torch.cat([full_pose, torch.zeros_like(full_pose[..., :6])], dim=1)
            flame_pose = full_pose[:, :-3].contiguous() if PER_FRAME_TRANSLATION else full_pose
            verts_p, landmarks2d_p, landmarks3d_p, landmarks2d_mediapipe = self.flame(shape_params=shape.expand(num_img, -1), expression_params=exp, pose_params=flame_pose[:,:9], eye_pose_params=flame_pose[:,9:])

            # Add per-frame translation
            if PER_FRAME_TRANSLATION:
                pose_translation = full_pose[:, -3:].unsqueeze(1)
                verts_p += pose_translation
                landmarks3d_p += pose_translation
                landmarks2d_p += pose_translation
                landmarks2d_mediapipe += pose_translation

            # perspective projection
            trans_landmarks2d_fan = projection(landmarks2d_p, cam_intrinsics, w2c)
            trans_landmarks2d_mp = projection(landmarks2d_mediapipe, cam_intrinsics, w2c)
            total_loss = 0.0
            # landmark loss
            landmark_loss2 = 0
            if USE_FAN_LANDMARKS:
                landmark_loss2 += lossfunc.l2_distance(trans_landmarks2d_fan[:, :landmarks_fan.shape[1], :2], landmarks_fan[:, :landmarks_fan.shape[1]])
            if USE_MP_LANDMARKS:
                # WARNING this is a L1 while the FAN loss is a L2 
                landmark_loss2 += lossfunc_mp.landmark_loss(trans_landmarks2d_mp, landmarks_mp)
            total_loss += landmark_loss2
            # regularizations
            total_loss += torch.mean(torch.square(shape)) * 0.1
            if self.optimize_expr:
                total_loss += torch.mean(torch.square(exp)) * 1e-2
            # penalize global rotation and neck rotation separately
            # encourage neck rotation (which was initialized at zero despite often being more likely than global rotation)
            # A potential improvement for this would be to enforce more temporal consistency on the global (shoulder) rotation
            total_loss += pose[..., 0:3].pow(2).mean() * 1e-1 * 2 # global rotation
            total_loss += pose[..., 3:6].pow(2).mean() * 1e-1 # neck rotation
            # temporal consistency
            total_loss += torch.mean(torch.square(pose[1:] - pose[:-1])) * 1
            if self.optimize_expr:
                total_loss += torch.mean(torch.square(exp[1:] - exp[:-1])) * 1e-1

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # visualize
            if k % 100 == 0:
                with torch.no_grad():
                    loss_info = "----iter: {}, time: {}\n".format(k,
                                                                  datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
                    loss_info = loss_info + f"landmark_loss: {landmark_loss2:.04f}\ttotal_loss: {total_loss:.04f}"
                    print(loss_info)
                    trans_verts = projection(verts_p[::visualize_sampling], cam_intrinsics, w2c[::visualize_sampling])
                    visdict = dict()
                    if USE_FAN_LANDMARKS:
                        visdict["landmarks_fan"] = tensor_vis_landmarks(visualize_images, landmarks_fan[::visualize_sampling], gt_landmarks=trans_landmarks2d_fan.detach()[::visualize_sampling])
                    if USE_MP_LANDMARKS:
                        visdict["landmarks_mp"] = tensor_vis_landmarks(visualize_images, landmarks_mp[::visualize_sampling], gt_landmarks=trans_landmarks2d_mp.detach()[::visualize_sampling])
                    visdict["shape_images"] = self.render.render_shape(verts_p[::visualize_sampling], trans_verts)
                    cv2.imwrite(os.path.join(savefolder, f"optimize_vis_{k:04d}.jpg"), visualize(visdict))

        if self.smooth:
            print("Smoothing tracking")
            conv_weights = gaussian_kernel(ksize=5, sigma=0.5).to(exp.device)
            exp = apply_featurewise_conv1d(exp, conv_weights, pad_mode="replicate")
            conv_weights = gaussian_kernel(ksize=7, sigma=2).to(exp.device)
            full_pose = apply_featurewise_conv1d(full_pose, conv_weights, pad_mode="replicate")

        save_intrinsics = [intrinsics[0] / size, intrinsics[1] / size, intrinsics[2] / size, intrinsics[3] / size]
        params = {}
        frames = []
        for i in range(num_img):
            frames.append({"file_path": "./crops/" + names[i],
                            "world_mat": w2c[i].detach().cpu().numpy().tolist(),
                            "expression": exp[i].detach().cpu().numpy().tolist(),
                            "pose": full_pose[i].detach().cpu().numpy().tolist(),
                            # "bbox": torch.stack(
                            #     [torch.min(landmark[i, :, 0]), torch.min(landmark[i, :, 1]),
                            #     torch.max(landmark[i, :, 0]), torch.max(landmark[i, :, 1])],
                            #     dim=0).detach().cpu().numpy().tolist(),
                            # "flame_keypoints": trans_landmarks2d[i, :,
                            #                     :2].detach().cpu().numpy().tolist()
                            })

        params["frames"] = frames
        params["intrinsics"] = save_intrinsics
        params["shape_params"] = shape[0].detach().cpu().numpy().tolist()
        with open(os.path.join(savefolder, save_name + ".json"), "w") as fp:
            json.dump(params, fp)

    def run(self, deca_code_file: str, lmk_fan_file: str, lmk_mp_file: str, lmk_iris_file: str, savefolder: str,
            intrinsics: List[float], save_name: str, iterations: int):
        device = self.device
        deca_code = json.load(open(deca_code_file, 'r'))
        landmarks_fan = torch.load(lmk_fan_file) if USE_FAN_LANDMARKS else None
        landmarks_mp = torch.load(lmk_mp_file) if USE_MP_LANDMARKS else None

        if os.path.exists(lmk_iris_file):
            use_iris = True
            print("Using iris keypoints")
            iris_kpts = torch.load(lmk_iris_file)
            if USE_FAN_LANDMARKS:
                landmarks_fan = torch.cat((landmarks_fan, iris_kpts[:, [1,0], :]), dim=1)
        else:
            use_iris = False
            print("Not using iris keypoints")

        if USE_FAN_LANDMARKS:
            landmarks_fan = landmarks_fan.float().to(device) * 2 / self.img_size - 1
        if USE_MP_LANDMARKS:
            landmarks_mp = landmarks_mp.float().to(device) * 2 / self.img_size - 1

        visualize_images, shape, exps, poses, cams, names = [], [], [], [], [], []
        num_img = len(deca_code)
        visualize_sampling = 1 if num_img < 24 else math.ceil(num_img / 24) # so there are 24 images visualized

        for k in range(1, num_img + 1):
            name = str(k)
            v = deca_code[name]   
            shape.append(torch.tensor(v["shape"]).float().to(device))
            exps.append(torch.tensor(v["exp"]).float().to(device))
            poses.append(torch.tensor(v["pose"]).float().to(device))
            cams.append(torch.tensor(v["cam"]).float().to(device))
            names.append(name)

        cat_fn = torch.stack if exps[0].ndim == 1 else torch.cat
        exps = cat_fn(exps, dim=0)
        poses = cat_fn(poses, dim=0)
        shape = cat_fn(shape, dim=0)
        cams = cat_fn(cams, dim=0)

        shape = shape.mean(dim=0).unsqueeze(0) if self.input_shape is None else self.input_shape

        # optimize
        self.optimize(shape, exps, landmarks_fan, landmarks_mp, poses, cams, names, visualize_images, visualize_sampling,
                      savefolder, intrinsics, save_name, use_iris, iterations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Landmarks-based facial pose, expression and shape optimization")
    parser.add_argument("--path", type=str, help="Path to images and deca and landmark jsons")
    parser.add_argument("--input_name", type=str, default="flame_params_inferred.json", help="Name for input json")
    parser.add_argument("--shape_from", type=str, default=".", help="Use shape parameter from this video if given.")
    parser.add_argument("--optimize_expr", action="store_true", help="Whether to optimize expression parameters.")
    parser.add_argument("--optimize_shape", action="store_true", help="Whether to optimize shape parameters.")
    parser.add_argument("--save_name", type=str, default="flame_params_optimized", help="Name for json")
    parser.add_argument("--fx", type=float, default=1500)
    parser.add_argument("--fy", type=float, default=1500)
    parser.add_argument("--cx", type=float, default=256)
    parser.add_argument("--cy", type=float, default=256)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--smooth", action="store_true", help="Apply a low-pass filter on the tracking results.")

    args = parser.parse_args()

    shape_from = None if args.shape_from in [None, "."] else args.shape_from
    image_path = os.path.join(args.path, "crops")
    model = Optimizer(image_path, shape_from, args.optimize_expr, args.optimize_shape, args.size, smooth=args.smooth)
    print("Optimizing: {}".format(args.path))

    intrinsics = [args.fx, args.fy, args.cx, args.cy]
    model.run(deca_code_file=os.path.join(args.path, args.input_name),
              lmk_fan_file=os.path.join(args.path, "landmarks_fan.pt"),
              lmk_mp_file=os.path.join(args.path, "landmarks_mp.pt"),
              lmk_iris_file=os.path.join(args.path, "landmarks_iris.pt"),
              savefolder=args.path, intrinsics=intrinsics,
              save_name=args.save_name, iterations=args.iterations)
