import torch
from torch import nn
import numpy as np
from utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        # Store as uint8 on CPU — 4x less memory than float32 on GPU.
        # Callers get a float32 GPU tensor via: cam.original_image.cuda().float().div_(255.0)
        self.original_image = image  # uint8 CHW CPU tensor
        self.image_width = image.shape[2]
        self.image_height = image.shape[1]

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        self.projection_matrix = (
            getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy)
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
