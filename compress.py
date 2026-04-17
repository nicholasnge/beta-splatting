import os
import argparse

from model import BetaModel


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compression script")
    p.add_argument("--ply", type=str, required=True)
    args = p.parse_args()

    print("Compressing " + args.ply)
    beta_model = BetaModel()
    beta_model.load_ply(args.ply)
    beta_model.save_png(os.path.dirname(args.ply))
