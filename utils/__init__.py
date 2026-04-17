"""
Shared utilities for beta-splatting.

Re-exports from sub-modules so callers can write:
    from utils import l1_loss, psnr, inverse_sigmoid, get_expon_lr_func, ...
"""
import sys
import random
from datetime import datetime
from errno import EEXIST
from os import makedirs, path
import os

import numpy as np
import torch

from utils.math import (
    inverse_sigmoid,
    strip_lowerdiag,
    strip_symmetric,
    build_rotation,
    build_scaling_rotation,
)
from utils.loss import l1_loss, l2_loss, ssim, psnr
from utils.image import PILtoTorch, colormap, apply_float_colormap, apply_depth_colormap
from utils.optimizer import get_expon_lr_func, cat_tensors_to_optimizer
from utils.camera import (
    BasicPointCloud,
    geom_transform_points,
    getWorld2View,
    getWorld2View2,
    getProjectionMatrix,
    fov2focal,
    focal2fov,
)
from utils.sh_utils import RGB2SH, SH2RGB


def mkdir_p(folder_path):
    try:
        makedirs(folder_path)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def searchForMaxIteration(folder):
    saved_iters = []
    for fname in os.listdir(folder):
        try:
            num = int(fname.split("_")[-1])
            saved_iters.append(num)
        except ValueError:
            continue
    return max(saved_iters) if saved_iters else None


def safe_state(silent=False):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(
                        x.replace(
                            "\n",
                            " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S"))),
                        )
                    )
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
