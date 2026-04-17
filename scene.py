import os
import random
import json
import torch
from tqdm import tqdm
from fused_ssim import fused_ssim
from lpipsPyTorch import lpips

import numpy as np
from dataset_readers import sceneLoadTypeCallbacks
from cameras import Camera
from utils import fov2focal, searchForMaxIteration, psnr


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = (
            int(orig_w / (resolution_scale * args.resolution) + 0.5),
            int(orig_h / (resolution_scale * args.resolution) + 0.5),
        )
    else:
        if args.resolution == -1:
            if orig_w > 1600:
                print(
                    "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                    "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                )
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    img = cam_info.image.resize(resolution)
    arr = np.array(img, dtype=np.uint8)

    # Apply alpha mask into RGB at load time so we don't need to carry it around
    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        rgb = arr[:, :, :3].astype(np.float32) * alpha
        arr = np.clip(rgb, 0, 255).astype(np.uint8)

    # Store as CHW uint8 on CPU — 4x less memory than float32 on GPU
    t = torch.from_numpy(arr)
    if t.dim() == 3:
        t = t.permute(2, 0, 1)[:3]  # HWC -> CHW, drop alpha if any
    elif t.dim() == 2:
        t = t.unsqueeze(0)

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=t,
        image_name=cam_info.image_name,
        uid=id,
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor() as pool:
        return list(pool.map(
            lambda x: loadCam(args, x[0], x[1], resolution_scale),
            enumerate(cam_infos)
        ))


def camera_to_JSON(id, camera):
    import numpy as np
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0
    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    return {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": [x.tolist() for x in rot],
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }


class Scene:
    def __init__(self, args, model, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.model = model
        self.best_psnr = 0

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval, init_type=args.init_type
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval
            )
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )

        if self.loaded_iter:
            self.model.load_ply(os.path.join(
                self.model_path, "point_cloud",
                "iteration_" + str(self.loaded_iter), "point_cloud.ply",
            ))
        else:
            self.model.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.model.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    @torch.no_grad()
    def save_best_model(self):
        from renderer import render
        psnr_test = 0.0
        test_view_stack = self.getTestCameras()
        for viewpoint in test_view_stack:
            image = torch.clamp(render(viewpoint, self.model)["render"], 0.0, 1.0)
            gt_image = viewpoint.original_image.cuda().float().div_(255.0)
            psnr_test += psnr(image, gt_image).mean()
        psnr_test /= len(test_view_stack)
        if psnr_test > self.best_psnr:
            self.save("best")
            self.best_psnr = psnr_test
            return True
        return False

    @torch.no_grad()
    def eval(self):
        from renderer import render
        torch.cuda.empty_cache()
        psnr_test = 0.0
        ssim_test = 0.0
        lpips_test = 0.0
        test_view_stack = self.getTestCameras()
        for viewpoint in tqdm(test_view_stack):
            image = torch.clamp(render(viewpoint, self.model)["render"], 0.0, 1.0)
            gt_image = viewpoint.original_image.cuda().float().div_(255.0)
            psnr_test += psnr(image, gt_image).mean()
            ssim_test += fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean()
            lpips_test += lpips(image, gt_image, net_type="vgg").mean()
        n = len(test_view_stack)
        result = {
            "PSNR": (psnr_test / n).item(),
            "SSIM": (ssim_test / n).item(),
            "LPIPS": (lpips_test / n).item(),
        }
        print(result)
        return result
