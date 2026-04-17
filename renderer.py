"""
Renderer: rasterize a BetaModel from a given camera.

render() is used during training and eval.
view() is used by the interactive viewer.
"""
import math
import torch
from torch.profiler import record_function
from gsplat.rendering import rasterization
from utils import apply_depth_colormap


def render(viewpoint_camera, model, render_mode="RGB", mask=None):
    """Rasterize model from viewpoint_camera. Returns a dict."""
    with record_function("render.build_K"):
        fx = 0.5 * viewpoint_camera.image_width / math.tan(viewpoint_camera.FoVx / 2)
        fy = 0.5 * viewpoint_camera.image_height / math.tan(viewpoint_camera.FoVy / 2)
        cx = viewpoint_camera.image_width / 2
        cy = viewpoint_camera.image_height / 2
        K = torch.tensor(
            [[fx, 0., cx], [0., fy, cy], [0., 0., 1.]],
            dtype=torch.float32, device="cuda",
        ).unsqueeze(0)  # [1, 3, 3]

    with record_function("render.gather"):
        if mask is not None:
            xyz      = model.get_xyz[mask]
            rotation = model.get_rotation[mask]
            scaling  = model.get_scaling[mask]
            opacity  = model.get_opacity.squeeze()[mask]
            beta     = model.get_beta.squeeze()[mask]
            shs      = model.get_shs[mask]
            sb_params = model.get_sb_params[mask]
        else:
            # No mask: pass all Gaussians directly — avoids nonzero syncs
            xyz      = model.get_xyz
            rotation = model.get_rotation
            scaling  = model.get_scaling
            opacity  = model.get_opacity.squeeze()
            beta     = model.get_beta.squeeze()
            shs      = model.get_shs
            sb_params = model.get_sb_params

    with record_function("render.rasterize"):
        rgbs, alphas, meta = rasterization(
            means=xyz,
            quats=rotation,
            scales=scaling,
            opacities=opacity,
            betas=beta,
            colors=shs,
            viewmats=viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0),
            Ks=K,
            width=viewpoint_camera.image_width,
            height=viewpoint_camera.image_height,
            backgrounds=model.background.unsqueeze(0),
            render_mode=render_mode,
            covars=None,
            sb_number=model.sb_number,
            sb_params=sb_params,
            packed=False,
        )

    rgbs = rgbs.permute(0, 3, 1, 2).contiguous()[0]
    return {
        "render": rgbs,
        "viewspace_points": meta["means2d"],
        "visibility_filter": meta["radii"] > 0,
        "radii": meta["radii"],
        "is_used": meta["radii"] > 0,
    }


@torch.no_grad()
def view(model, camera_state, render_tab_state, center=None):
    """Callable for the interactive viewer."""
    from viewer import BetaRenderTabState
    assert isinstance(render_tab_state, BetaRenderTabState)

    if render_tab_state.preview_render:
        W = render_tab_state.render_width
        H = render_tab_state.render_height
    else:
        W = render_tab_state.viewer_width
        H = render_tab_state.viewer_height

    c2w = camera_state.c2w
    K = camera_state.get_K((W, H))
    c2w = torch.from_numpy(c2w).float().to("cuda")
    K = torch.from_numpy(K).float().to("cuda")

    if center:
        xyz = model._xyz - model._xyz.mean(dim=0, keepdim=True)
    else:
        xyz = model._xyz

    render_mode = render_tab_state.render_mode
    mask = torch.logical_and(
        model._beta >= render_tab_state.b_range[0],
        model._beta <= render_tab_state.b_range[1],
    ).squeeze()
    model.background = torch.tensor(render_tab_state.backgrounds, device="cuda") / 255.0

    render_colors, alphas, meta = rasterization(
        means=xyz[mask],
        quats=model.get_rotation[mask],
        scales=model.get_scaling[mask],
        opacities=model.get_opacity.squeeze()[mask],
        betas=model.get_beta.squeeze()[mask],
        colors=model.get_shs[mask],
        viewmats=torch.linalg.inv(c2w).unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=W,
        height=H,
        backgrounds=model.background.unsqueeze(0),
        render_mode=render_mode if render_mode != "Alpha" else "RGB",
        covars=None,
        sb_number=model.sb_number,
        sb_params=model.get_sb_params[mask],
        packed=False,
        near_plane=render_tab_state.near_plane,
        far_plane=render_tab_state.far_plane,
        radius_clip=render_tab_state.radius_clip,
    )
    render_tab_state.total_count_number = len(model.get_xyz)
    render_tab_state.rendered_count_number = (meta["radii"] > 0).sum().item()

    if render_mode == "Alpha":
        render_colors = alphas
    if render_colors.shape[-1] == 1:
        render_colors = apply_depth_colormap(render_colors)

    return render_colors[0].clamp(0, 1).cpu().numpy()
