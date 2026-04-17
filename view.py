import time
import argparse
import torch
import viser

from model import BetaModel
from renderer import view
from viewer import BetaViewer


def parse_args():
    p = argparse.ArgumentParser(description="Viewer script")
    p.add_argument("--ply", type=str, default=None)
    p.add_argument("--png", type=str, default=None)
    p.add_argument("--sh_degree", type=int, default=0)
    p.add_argument("--sb_number", type=int, default=2)
    p.add_argument("--white_background", "-w", action="store_true")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--share_url", action="store_true")
    p.add_argument("--center", action="store_true")
    return p.parse_args()


@torch.no_grad()
def viewing(args):
    beta_model = BetaModel(args.sh_degree, args.sb_number)
    if args.ply:
        beta_model.load_ply(args.ply)
    elif args.png:
        beta_model.load_png(args.png)
    else:
        raise ValueError("You must provide either a .ply file or a .png folder")

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    beta_model.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    server = viser.ViserServer(port=args.port, verbose=False)
    viewer = BetaViewer(
        server=server,
        render_fn=lambda camera_state, render_tab_state: view(
            beta_model, camera_state, render_tab_state, args.center
        ),
        mode="rendering",
        share_url=args.share_url,
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    args = parse_args()
    viewing(args)
