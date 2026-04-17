import os
import sys
import time
import json
import argparse
import contextlib
from random import randint

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
from fused_ssim import fused_ssim

from utils import l1_loss, safe_state
from model import BetaModel
from scene import Scene
from renderer import render, view
from utils.math import build_scaling_rotation
import viser
from viewer import BetaViewer


class TrainingProfiler:
    """Profiles training iterations using two complementary PyTorch tools:

    1. PyTorch profiler  — records how long each named section takes on CPU and
       GPU, and how much GPU memory each op allocates. Runs for a fixed number
       of iterations then writes two outputs:
         - trace.json      : open at chrome://tracing for a per-section timeline
         - console table   : printed immediately, ranked by CUDA time and memory

    2. GPU memory snapshot — records every GPU allocation and free with a full
       Python stack trace for the entire profiled window. Written as:
         - memory_snapshot.pkl : drag to https://pytorch.org/memory_viz for a
                                 flame graph showing which line of code owns
                                 how much VRAM at any point in time

    Usage in the training loop:
        profiler.start()
        for iteration in loop:
            with profiler.record("section_name"):
                ...do work...
            profiler.step()   # advance internal schedule; auto-stops after
                              # profile_iters iterations
        profiler.stop()       # no-op if already stopped; also saves snapshot
    """

    def __init__(self, output_dir, enabled=False, profile_iters=5, profile_from=0):
        self.enabled = enabled
        self.output_dir = output_dir
        self.profile_iters = profile_iters
        self.profile_from = profile_from
        self._profiler = None
        self._iter = 0
        self._started = False

    def start(self):
        """Start profiling. If profile_from > 0, call this when that iteration is reached."""
        if not self.enabled or self._started:
            return
        self._started = True
        os.makedirs(self.output_dir, exist_ok=True)

        torch.cuda.memory._record_memory_history(max_entries=100_000)

        self._profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=self.profile_iters),
            on_trace_ready=self._on_trace_ready,
        )
        self._profiler.start()
        print(f"[Profiler] Started at iteration {self.profile_from} — 1 warmup + {self.profile_iters} active")
        print(f"[Profiler] Outputs will be written to: {self.output_dir}")

    def record(self, label):
        """Context manager that annotates a block with a label in the trace.
        When profiling is disabled this is a no-op, so there is zero overhead."""
        if self.enabled:
            return record_function(label)
        return contextlib.nullcontext()

    def step(self):
        """Advance the profiler by one iteration. Auto-stops after the active
        window completes so the rest of training runs without overhead."""
        if not self.enabled or self._profiler is None:
            return
        self._profiler.step()
        self._iter += 1
        if self._iter >= 1 + self.profile_iters:  # warmup + active
            self.stop()

    def stop(self):
        """Stop profiling and write the memory snapshot. Safe to call twice."""
        if not self.enabled:
            return
        if self._profiler is not None:
            self._profiler.stop()
            self._profiler = None
        snap_path = os.path.join(self.output_dir, "memory_snapshot.pkl")
        torch.cuda.memory._dump_snapshot(snap_path)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"[Profiler] Memory snapshot → {snap_path}")
        print(f"[Profiler] Visualize at https://pytorch.org/memory_viz (drag and drop)")
        self.enabled = False  # prevent double-stop

    def _on_trace_ready(self, prof):
        """Called automatically by the profiler once the active window ends."""
        trace_path = os.path.join(self.output_dir, "trace.json")
        prof.export_chrome_trace(trace_path)
        print(f"\n[Profiler] Chrome trace → {trace_path}")
        print(f"[Profiler] Open at chrome://tracing (load the file)\n")
        print("[Profiler] Top sections by CUDA time:")
        print(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=15, max_name_column_width=40))
        print("\n[Profiler] Top sections by GPU memory allocated:")
        print(prof.key_averages().table(
            sort_by="self_cuda_memory_usage", row_limit=15, max_name_column_width=40))


def parse_args():
    p = argparse.ArgumentParser(description="Beta Splatting Training")

    # Data
    p.add_argument("--source_path", "-s", required=True)
    p.add_argument("--model_path", "-m", default="")
    p.add_argument("--images", "-i", default="images")
    p.add_argument("--resolution", "-r", type=int, default=-1)
    p.add_argument("--white_background", "-w", action="store_true")
    p.add_argument("--data_device", default="cuda")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--cap_max", type=int, default=1000000)
    p.add_argument("--init_type", default="sfm", choices=["sfm", "random"])

    # Model
    p.add_argument("--sh_degree", type=int, default=0)
    p.add_argument("--sb_number", type=int, default=2)

    # Optimization
    p.add_argument("--iterations", type=int, default=30000)
    p.add_argument("--position_lr_init", type=float, default=0.00016)
    p.add_argument("--position_lr_final", type=float, default=0.0000016)
    p.add_argument("--position_lr_delay_mult", type=float, default=0.01)
    p.add_argument("--position_lr_max_steps", type=int, default=30000)
    p.add_argument("--sh_lr", type=float, default=0.0025)
    p.add_argument("--sb_params_lr", type=float, default=0.0025)
    p.add_argument("--opacity_lr", type=float, default=0.05)
    p.add_argument("--beta_lr", type=float, default=0.001)
    p.add_argument("--scaling_lr", type=float, default=0.005)
    p.add_argument("--rotation_lr", type=float, default=0.001)
    p.add_argument("--lambda_dssim", type=float, default=0.2)
    p.add_argument("--densification_interval", type=int, default=100)
    p.add_argument("--densify_from_iter", type=int, default=500)
    p.add_argument("--densify_until_iter", type=int, default=25000)
    p.add_argument("--random_background", action="store_true")
    p.add_argument("--noise_lr", type=float, default=5e4)
    p.add_argument("--scale_reg", type=float, default=0.01)
    p.add_argument("--opacity_reg", type=float, default=0.01)

    # Checkpoints
    p.add_argument("--save_iterations", nargs="+", type=int, default=[])
    p.add_argument("--start_checkpoint", type=str, default=None)

    # Viewer
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--disable_viewer", action="store_true")
    p.add_argument("--share_url", action="store_true")
    p.add_argument("--center", action="store_true")

    # Misc
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-compress", action="store_true", dest="no_compress", help="Disable PNG compression at end")

    # Profiling
    p.add_argument("--profile", action="store_true", help="Enable torch profiler (trace.json + memory_snapshot.pkl)")
    p.add_argument("--profile_iters", type=int, default=5, help="Number of iterations to profile")
    p.add_argument("--profile_from", type=int, default=0, help="Iteration to start profiling (0=immediately)")

    return p.parse_args()


def prepare_output(args):
    if not args.model_path:
        args.model_path = os.path.join("./output/", os.path.basename(args.source_path))
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(vars(args)))


def training(args):
    first_iter = 0
    prepare_output(args)

    beta_model = BetaModel(args.sh_degree, args.sb_number)
    scene = Scene(args, beta_model)
    beta_model.training_setup(args)

    if args.start_checkpoint:
        (model_params, first_iter) = torch.load(args.start_checkpoint)
        beta_model.restore(model_params, args)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    profiler = TrainingProfiler(
        output_dir=args.model_path,
        enabled=args.profile,
        profile_iters=args.profile_iters,
        profile_from=args.profile_from)
    if args.profile_from == 0:
        profiler.start()

    if not args.disable_viewer:
        server = viser.ViserServer(port=args.port, verbose=False)
        viewer = BetaViewer(
            server=server,
            render_fn=lambda camera_state, render_tab_state: view(
                beta_model, camera_state, render_tab_state, args.center
            ),
            mode="training",
            share_url=args.share_url,
        )

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    patience = 20
    patience_counter = 0

    iteration = first_iter + 1
    if args.cap_max < beta_model._xyz.shape[0]:
        print(
            f"Warning: cap_max ({args.cap_max}) is smaller than the number of points initialized "
            f"({beta_model._xyz.shape[0]}). Resetting cap_max to the number of points initialized."
        )
        args.cap_max = beta_model._xyz.shape[0]

    if not args.eval:
        progress_bar = tqdm(range(first_iter, args.iterations), desc="Training progress")
    else:
        progress_bar = tqdm(desc="Training progress")

    while True:
        if not args.eval and iteration > args.iterations:
            break

        torch.cuda.reset_peak_memory_stats()

        if args.profile and iteration == args.profile_from:
            profiler.start()

        if not args.disable_viewer:
            while viewer.state == "paused":
                time.sleep(0.01)
            viewer.lock.acquire()
            tic = time.time()

        xyz_lr = beta_model.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            beta_model.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        beta_model.background = (
            torch.rand((3), device="cuda") if args.random_background else background
        )
        with profiler.record("render"):
            render_pkg = render(viewpoint_cam, beta_model)
        image = render_pkg["render"]

        with profiler.record("gt_upload"):
            gt_image = viewpoint_cam.original_image.cuda().float().div_(255.0)
        with profiler.record("loss"):
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (
                1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            )
            if args.densify_from_iter < iteration < args.densify_until_iter:
                loss += args.opacity_reg * torch.abs(beta_model.get_opacity).mean()
                loss += args.scale_reg * torch.abs(beta_model.get_scaling).mean()

        with profiler.record("backward"):
            loss.backward()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration in args.save_iterations:
                print(f"\n[ITER {iteration}] Saving model")
                scene.save(iteration)

            if (
                iteration < args.densify_until_iter
                and iteration > args.densify_from_iter
                and iteration % args.densification_interval == 0
            ):
                with profiler.record("densify"):
                    dead_mask = (beta_model.get_opacity <= 0.005).squeeze(-1)
                    beta_model.relocate_gs(dead_mask=dead_mask)
                    beta_model.add_new_gs(cap_max=args.cap_max)

                    L = build_scaling_rotation(beta_model.get_scaling, beta_model.get_rotation)
                    actual_covariance = L @ L.transpose(1, 2)
                    noise = (
                        torch.randn_like(beta_model._xyz)
                        * (torch.pow(1 - beta_model.get_opacity, 100))
                        * args.noise_lr
                        * xyz_lr
                    )
                    noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                    beta_model._xyz.add_(noise)

            with profiler.record("optimizer"):
                beta_model.optimizer.step()
                beta_model.optimizer.zero_grad(set_to_none=True)

            if iteration % 100 == 0:
                vram_mb = torch.cuda.max_memory_allocated() / 1024**2
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.4f}",
                    "Beta": f"{beta_model._beta.mean().item():.2f}",
                    "#gs": beta_model._xyz.shape[0],
                    "VRAM": f"{vram_mb:.0f}MiB",
                })
            else:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.7f}",
                    "Beta": f"{beta_model._beta.mean().item():.2f}",
                })
            progress_bar.update(1)

            if not args.disable_viewer:
                num_train_rays_per_step = gt_image.numel()
                viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic + 1e-8)
                viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                viewer.update(iteration, num_train_rays_per_step)

            if args.eval and iteration % 500 == 0 and iteration >= 15_000:
                if scene.save_best_model():
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping.")
                    break

        profiler.step()
        iteration += 1

    profiler.stop()
    progress_bar.close()
    print("\nTraining complete.\n")

    if args.eval:
        print("\nEvaluating Best Model Performance\n")
        beta_model.load_ply(
            os.path.join(scene.model_path, "point_cloud/iteration_best/point_cloud.ply")
        )
        result = scene.eval()
        with open(
            os.path.join(scene.model_path, "point_cloud/iteration_best/metrics.json"), "w"
        ) as f:
            json.dump(result, f, indent=True)

    if not args.no_compress:
        if args.eval:
            print("Compressing model at iteration_best...")
            beta_model.save_png(os.path.join(scene.model_path, "point_cloud/iteration_best"))
        else:
            last_iter = args.save_iterations[-1]
            print(f"Compressing model at iteration {last_iter}...")
            beta_model.save_png(os.path.join(scene.model_path, f"point_cloud/iteration_{last_iter}"))

    if not args.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    args = parse_args()
    args.save_iterations.append(args.iterations)
    args.source_path = os.path.abspath(args.source_path)
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    training(args)
