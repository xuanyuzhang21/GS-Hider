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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import torch.nn as nn
from torch.nn import functional as F
from decoder import SimpleCNN, WatermarkCNN
import time


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    imagenet = SimpleCNN().cuda()
    imagenet.load_state_dict(torch.load('/userhome/water/3DGS_single/output/stega_bicycle_0918/chkpnt50000_net.pth'))
    imagenet.eval()

    waternet = WatermarkCNN().cuda()
    waternet.load_state_dict(torch.load('/userhome/water/3DGS_single/output/stega_bicycle_0918/chkpnt50000_waternet.pth'))
    waternet.eval()   

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    count = 0

    rendering_time = 0

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()
        start = time.time()
        rendering = render(view, gaussians, pipeline, background)["render"]

        water = waternet(rendering)
        rendering = imagenet(rendering)
        torch.cuda.synchronize()
        end = time.time()
        rendering_time += end - start
        count += 1
        gt = view.original_image[0:3, :, :]
        gt_wm = view.watermark_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(water, os.path.join(render_path, '{0:05d}_water'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt_wm, os.path.join(gts_path, '{0:05d}_water'.format(idx) + ".png"))
    print(rendering_time / count)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)