"""
Render test cameras from a trained BetaModel checkpoint.
Saves rendered images, GT images, and a metrics JSON.
"""
import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from fused_ssim import fused_ssim
from lpipsPyTorch import lpips

from model import BetaModel
from scene import Scene
from renderer import render
from utils import psnr


def parse_args():
    p = argparse.ArgumentParser(description="Render and evaluate a trained BetaModel")
    p.add_argument("--source_path", "-s", required=True)
    p.add_argument("--model_path", "-m", required=True)
    p.add_argument("--iteration", "-i", default="30000", type=str)
    p.add_argument("--resolution", "-r", type=int, default=-1)
    p.add_argument("--images", default="images")
    p.add_argument("--white_background", "-w", action="store_true")
    p.add_argument("--sh_degree", type=int, default=0)
    p.add_argument("--sb_number", type=int, default=2)
    p.add_argument("--init_type", default="sfm")
    p.add_argument("--out_dir", default=None, help="Override output directory")
    return p.parse_args()


def tensor_to_pil(t):
    """[C,H,W] float32 in [0,1] → PIL image."""
    arr = (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


@torch.no_grad()
def main(args):
    args.eval = True
    args.data_device = "cuda"
    args.source_path = os.path.abspath(args.source_path)

    out_dir = args.out_dir or os.path.join(
        args.model_path, "renders", f"iteration_{args.iteration}"
    )
    renders_dir = os.path.join(out_dir, "renders")
    gt_dir = os.path.join(out_dir, "gt")
    os.makedirs(renders_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]

    model = BetaModel(args.sh_degree, args.sb_number)
    model.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    scene = Scene(args, model)

    ply_path = os.path.join(
        args.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply"
    )
    print(f"Loading {ply_path}")
    model.load_ply(ply_path)

    test_cams = scene.getTestCameras()
    print(f"Rendering {len(test_cams)} test views at {test_cams[0].image_width}x{test_cams[0].image_height}")

    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for cam in tqdm(test_cams):
        rendered = render(cam, model)["render"].clamp(0, 1)
        gt = cam.original_image.cuda().float().div_(255.0)

        psnr_v = psnr(rendered.unsqueeze(0), gt.unsqueeze(0)).mean().item()
        ssim_v = fused_ssim(rendered.unsqueeze(0), gt.unsqueeze(0)).item()
        lpips_v = lpips(rendered, gt, net_type="vgg").item()

        psnr_vals.append(psnr_v)
        ssim_vals.append(ssim_v)
        lpips_vals.append(lpips_v)

        tensor_to_pil(rendered).save(os.path.join(renders_dir, f"{cam.image_name}.png"))
        tensor_to_pil(gt).save(os.path.join(gt_dir, f"{cam.image_name}.png"))

    metrics = {
        "PSNR":  round(float(np.mean(psnr_vals)),  4),
        "SSIM":  round(float(np.mean(ssim_vals)),  4),
        "LPIPS": round(float(np.mean(lpips_vals)), 4),
        "n_views": len(test_cams),
        "per_view": {
            cam.image_name: {"PSNR": round(p, 4), "SSIM": round(s, 4), "LPIPS": round(l, 4)}
            for cam, p, s, l in zip(test_cams, psnr_vals, ssim_vals, lpips_vals)
        },
    }

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*40}")
    print(f"  PSNR:  {metrics['PSNR']:.2f} dB")
    print(f"  SSIM:  {metrics['SSIM']:.4f}")
    print(f"  LPIPS: {metrics['LPIPS']:.4f}")
    print(f"  Views: {metrics['n_views']}")
    print(f"{'='*40}")
    print(f"Renders → {renders_dir}")
    print(f"Metrics → {metrics_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
