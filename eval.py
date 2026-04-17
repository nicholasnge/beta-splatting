import os
import sys
import argparse
import torch

from model import BetaModel
from scene import Scene


def parse_args():
    p = argparse.ArgumentParser(description="Evaluation script")
    p.add_argument("--source_path", "-s", required=True)
    p.add_argument("--model_path", "-m", required=True)
    p.add_argument("--images", "-i", default="images")
    p.add_argument("--resolution", "-r", type=int, default=-1)
    p.add_argument("--white_background", "-w", action="store_true")
    p.add_argument("--data_device", default="cuda")
    p.add_argument("--sh_degree", type=int, default=0)
    p.add_argument("--sb_number", type=int, default=2)
    p.add_argument("--init_type", default="sfm")
    p.add_argument("--iteration", default="best", type=str)
    return p.parse_args()


def main(args):
    args.eval = True
    beta_model = BetaModel(args.sh_degree, args.sb_number)
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    beta_model.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    scene = Scene(args, beta_model)

    ply_path = os.path.join(args.model_path, "point_cloud", "iteration_" + args.iteration, "point_cloud.ply")
    if os.path.exists(ply_path):
        print("Evaluating " + ply_path)
        beta_model.load_ply(ply_path)
        scene.eval()

    png_path = os.path.join(args.model_path, "point_cloud", "iteration_" + args.iteration, "png")
    if os.path.exists(png_path):
        print("Evaluating " + png_path)
        beta_model.load_png(png_path)
        scene.eval()


if __name__ == "__main__":
    args = parse_args()
    args.source_path = os.path.abspath(args.source_path)
    print("Evaluating " + args.model_path)
    main(args)
