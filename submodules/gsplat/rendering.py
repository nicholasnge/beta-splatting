import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Literal

from .cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
    spherical_beta,
)
from .utils import depth_to_normal, get_projection_matrix


def rasterization(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    betas: Tensor,  # [N]
    colors: Tensor,  # [(C,) N, K, 3] — DC SH coefficients
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sb_number: Optional[int] = None,
    sb_params: Optional[Tensor] = None,  # [N, sb_number, 6]
    packed: bool = True,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal[
        "RGB", "Normal", "Diffuse", "Specular", "Depth", "EDepth", "RGB+D", "RGB+ED"
    ] = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    ortho: bool = False,
    covars: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Dict]:
    """Rasterize a set of 3D Gaussians (N) to a batch of image planes (C).

    This function provides a handful features for 3D Gaussian rasterization, which
    we detail in the following notes. A complete profiling of the these features
    can be found in the :ref:`profiling` page.

    .. note::
        **Multi-GPU Distributed Rasterization**: This function can be used in a multi-GPU
        distributed scenario by setting `distributed` to True. When `distributed` is True,
        a subset of total Gaussians could be passed into this function in each rank, and
        the function will collaboratively render a set of images using Gaussians from all ranks. Note
        to achieve balanced computation, it is recommended (not enforced) to have similar number of
        Gaussians in each rank. But we do enforce that the number of cameras to be rendered
        in each rank is the same. The function will return the rendered images
        corresponds to the input cameras in each rank, and allows for gradients to flow back to the
        Gaussians living in other ranks. For the details, please refer to the paper
        `On Scaling Up 3D Gaussian Splatting Training <https://arxiv.org/abs/2406.18533>`_.

    .. note::
        **Batch Rasterization**: This function allows for rasterizing a set of 3D Gaussians
        to a batch of images in one go, by simplly providing the batched `viewmats` and `Ks`.

    .. note::
        **Support N-D Features**: If `view_dependent_param_count` is None,
        the `colors` is expected to be with shape [N, D] or [C, N, D], in which D is the channel of
        the features to be rendered. The computation is slow when D > 32 at the moment.
        If `view_dependent_param_count` is set, the `colors` is expected to be the SH coefficients with
        shape [N, K, 3] or [C, N, K, 3], where K is the number of SH bases. In this case, it is expected
        that :math:`(\\textit{view_dependent_param_count} + 1) ^ 2 \\leq K`, where `view_dependent_param_count` controls the
        activated bases in the SH coefficients.

    .. note::
        **Depth Rendering**: This function supports colors or/and depths via `render_mode`.
        The supported modes are "RGB", "D", "ED", "RGB+D", and "RGB+ED". "RGB" renders the
        colored image that respects the `colors` argument. "D" renders the accumulated z-depth
        :math:`\\sum_i w_i z_i`. "ED" renders the expected z-depth
        :math:`\\frac{\\sum_i w_i z_i}{\\sum_i w_i}`. "RGB+D" and "RGB+ED" render both
        the colored image and the depth, in which the depth is the last channel of the output.

    .. note::
        **Memory-Speed Trade-off**: The `packed` argument provides a trade-off between
        memory footprint and runtime. If `packed` is True, the intermediate results are
        packed into sparse tensors, which is more memory efficient but might be slightly
        slower. This is especially helpful when the scene is large and each camera sees only
        a small portion of the scene. If `packed` is False, the intermediate results are
        with shape [C, N, ...], which is faster but might consume more memory.

    .. note::
        **Sparse Gradients**: If `sparse_grad` is True, the gradients for {means, quats, scales}
        will be stored in a `COO sparse layout <https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html>`_.
        This can be helpful for saving memory
        for training when the scene is large and each iteration only activates a small portion
        of the Gaussians. Usually a sparse optimizer is required to work with sparse gradients,
        such as `torch.optim.SparseAdam <https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html#sparseadam>`_.
        This argument is only effective when `packed` is True.

    .. note::
        **Speed-up for Large Scenes**: The `radius_clip` argument is extremely helpful for
        speeding up large scale scenes or scenes with large depth of fields. Gaussians with
        2D radius smaller or equal than this value (in pixel unit) will be skipped during rasterization.
        This will skip all the far-away Gaussians that are too small to be seen in the image.
        But be warned that if there are close-up Gaussians that are also below this threshold, they will
        also get skipped (which is rarely happened in practice). This is by default disabled by setting
        `radius_clip` to 0.0.

    .. note::
        **Antialiased Rendering**: If `rasterize_mode` is "antialiased", the function will
        apply a view-dependent compensation factor
        :math:`\\rho=\\sqrt{\\frac{Det(\\Sigma)}{Det(\\Sigma+ \\epsilon I)}}` to Gaussian
        opacities, where :math:`\\Sigma` is the projected 2D covariance matrix and :math:`\\epsilon`
        is the `eps2d`. This will make the rendered image more antialiased, as proposed in
        the paper `Mip-Splatting: Alias-free 3D Gaussian Splatting <https://arxiv.org/pdf/2311.16493>`_.

    .. note::
        **AbsGrad**: If `absgrad` is True, the absolute gradients of the projected
        2D means will be computed during the backward pass, which could be accessed by
        `meta["means2d"].absgrad`. This is an implementation of the paper
        `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_,
        which is shown to be more effective for splitting Gaussians during training.

    .. warning::
        This function is currently not differentiable w.r.t. the camera intrinsics `Ks`.

    Args:
        means: The 3D centers of the Gaussians. [N, 3]
        quats: The quaternions of the Gaussians (wxyz convension). It's not required to be normalized. [N, 4]
        scales: The scales of the Gaussians. [N, 3]
        opacities: The opacities of the Gaussians. [N]
        betas: The sharpness of the Kernels. [N]
        colors: The colors of the Gaussians. [(C,) N, D] or [(C,) N, K, 3] for SH coefficients.
        viewmats: The world-to-cam transformation of the cameras. [C, 4, 4]
        Ks: The camera intrinsics. [C, 3, 3]
        width: The width of the image.
        height: The height of the image.
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        radius_clip: Gaussians with 2D radius smaller or equal than this value will be
            skipped. This is extremely helpful for speeding up large scale scenes.
            Default is 0.0.
        eps2d: An epsilon added to the egienvalues of projected 2D covariance matrices.
            This will prevents the projected GS to be too small. For example eps2d=0.3
            leads to minimal 3 pixel unit. Default is 0.3.
        view_dependent_param_count: The number of SH bands to use or the number of
            active primitives for spherical beta. Default is None.
        packed: Whether to use packed mode which is more memory efficient but might or
            might not be as fast. Default is True.
        tile_size: The size of the tiles for rasterization. Default is 16.
            (Note: other values are not tested)
        backgrounds: The background colors. [C, D]. Default is None.
        render_mode: The rendering mode. Supported modes are "RGB", "D", "ED", "RGB+D",
            and "RGB+ED". "RGB" renders the colored image, "D" renders the accumulated depth, and
            "ED" renders the expected depth. Default is "RGB".
        sparse_grad: If true, the gradients for {means, quats, scales} will be stored in
            a COO sparse layout. This can be helpful for saving memory. Default is False.
        absgrad: If true, the absolute gradients of the projected 2D means
            will be computed during the backward pass, which could be accessed by
            `meta["means2d"].absgrad`. Default is False.
        rasterize_mode: The rasterization mode. Supported modes are "classic" and
            "antialiased". Default is "classic".
        channel_chunk: The number of channels to render in one go. Default is 32.
            If the required rendering channels are larger than this value, the rendering
            will be done looply in chunks.
        distributed: Whether to use distributed rendering. Default is False. If True,
            The input Gaussians are expected to be a subset of scene in each rank, and
            the function will collaboratively render the images for all ranks.
        ortho: Whether to use orthographic projection. In such case fx and fy become the scaling
            factors to convert projected coordinates into pixel space and cx, cy become offsets.
        covars: Optional covariance matrices of the Gaussians. If provided, the `quats` and
            `scales` will be ignored. [N, 3, 3], Default is None.

    Returns:
        A tuple:

        **render_colors**: The rendered colors. [C, height, width, X].
        X depends on the `render_mode` and input `colors`. If `render_mode` is "RGB",
        X is D; if `render_mode` is "D" or "ED", X is 1; if `render_mode` is "RGB+D" or
        "RGB+ED", X is D+1.

        **render_alphas**: The rendered alphas. [C, height, width, 1].

        **meta**: A dictionary of intermediate results of the rasterization.

    """
    meta = {}

    N = means.shape[0]
    C = viewmats.shape[0]
    device = means.device
    assert means.shape == (N, 3), means.shape
    if covars is None:
        assert quats.shape == (N, 4), quats.shape
        assert scales.shape == (N, 3), scales.shape
    else:
        assert covars.shape == (N, 3, 3), covars.shape
        quats, scales = None, None
        # convert covars from 3x3 matrix to upper-triangular 6D vector
        tri_indices = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
        covars = covars[..., tri_indices[0], tri_indices[1]]
    assert opacities.shape == (N,), opacities.shape
    assert betas.shape == (N,), betas.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape
    assert render_mode in [
        "RGB",
        "Normal",
        "Diffuse",
        "Specular",
        "Depth",
        "EDepth",
        "RGB+D",
        "RGB+ED",
    ], render_mode

    # colors: [N, K, 3] or [C, N, K, 3] (DC SH coefficients, K >= 1)
    assert (colors.dim() == 3 and colors.shape[0] == N and colors.shape[2] == 3) or (
        colors.dim() == 4 and colors.shape[:2] == (C, N) and colors.shape[3] == 3
    ), colors.shape
    if sb_number:
        assert sb_params.shape == (C, N, sb_number, 6) or sb_params.shape == (
            N,
            sb_number,
            6,
        ), sb_params.shape

    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    proj_results = fully_fused_projection(
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        packed=packed,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=sparse_grad,
        calc_compensations=(rasterize_mode == "antialiased"),
        ortho=ortho,
    )

    if packed:
        # The results are packed into shape [nnz, ...]. All elements are valid.
        (
            camera_ids,
            primitive_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        ) = proj_results
        opacities = opacities[primitive_ids]  # [nnz]
        betas = betas[primitive_ids]  # [nnz]
    else:
        # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
        radii, means2d, depths, conics, compensations = proj_results
        opacities = opacities.repeat(C, 1)  # [C, N]
        betas = betas.repeat(C, 1)  # [C, N]
        camera_ids, primitive_ids = None, None

    if compensations is not None:
        opacities = opacities * compensations

    meta.update(
        {
            # global camera_ids
            "camera_ids": camera_ids,
            # local primitive_ids
            "primitive_ids": primitive_ids,
            "radii": radii,
            "means2d": means2d,
            "depths": depths,
            "conics": conics,
            "opacities": opacities,
            "betas": betas,
        }
    )

    # Evaluate degree-0 SH to get base RGB, then apply spherical beta.
    # colors: [N, K, 3] or [C, N, K, 3] — only the DC term (index 0) is used.
    SH_C0 = 0.28209479177387814
    camtoworlds = torch.inverse(viewmats)  # [C, 4, 4]
    if packed:
        dirs = means[primitive_ids, :] - camtoworlds[camera_ids, :3, 3]  # [nnz, 3]
        masks = radii > 0  # [nnz]
        shs = colors[primitive_ids] if colors.dim() == 3 else colors[camera_ids, primitive_ids]  # [nnz, K, 3]
    else:
        dirs = means[None, :, :] - camtoworlds[:, None, :3, 3]  # [C, N, 3]
        masks = radii > 0  # [C, N]
        shs = colors.expand(C, -1, -1, -1) if colors.dim() == 3 else colors  # [C, N, K, 3]
    colors = torch.clamp_min(shs[..., 0, :] * SH_C0 + 0.5, 0.0)
    if sb_number:
        if sb_params.dim() == 3:
            sb_params = sb_params.expand(C, -1, -1, -1)
        if render_mode == "Diffuse":
            sb_params = torch.cat(
                [torch.zeros_like(sb_params[..., :3]), sb_params.clone()[..., 3:]], -1
            )
        colors = spherical_beta(sb_number, dirs, colors, sb_params, masks=masks)
        colors = torch.clamp_min(colors, 0.0)

    # Rasterize to pixels
    if render_mode in ["RGB+D", "RGB+ED"]:
        colors = torch.cat((colors, depths[..., None]), dim=-1)
        if backgrounds is not None:
            backgrounds = torch.cat(
                [backgrounds, torch.zeros(C, 1, device=backgrounds.device)], dim=-1
            )
    elif render_mode in ["Depth", "EDepth", "Normal"]:
        colors = depths[..., None]
        if backgrounds is not None:
            backgrounds = torch.zeros(C, 1, device=backgrounds.device)
    else:  # RGB
        pass

    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_cameras=C,
        camera_ids=camera_ids,
        primitive_ids=primitive_ids,
    )
    # print("rank", world_rank, "Before isect_offset_encode")
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    meta.update(
        {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "tiles_per_gauss": tiles_per_gauss,
            "isect_ids": isect_ids,
            "flatten_ids": flatten_ids,
            "isect_offsets": isect_offsets,
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "n_cameras": C,
        }
    )

    # print("rank", world_rank, "Before rasterize_to_pixels")
    if colors.shape[-1] > channel_chunk:
        # slice into chunks
        n_chunks = (colors.shape[-1] + channel_chunk - 1) // channel_chunk
        render_colors, render_alphas = [], []
        for i in range(n_chunks):
            colors_chunk = colors[..., i * channel_chunk : (i + 1) * channel_chunk]
            backgrounds_chunk = (
                backgrounds[..., i * channel_chunk : (i + 1) * channel_chunk]
                if backgrounds is not None
                else None
            )
            render_colors_, render_alphas_ = rasterize_to_pixels(
                means2d,
                conics,
                colors_chunk,
                opacities,
                betas,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds_chunk,
                packed=packed,
                absgrad=absgrad,
            )
            render_colors.append(render_colors_)
            render_alphas.append(render_alphas_)
        render_colors = torch.cat(render_colors, dim=-1)
        render_alphas = render_alphas[0]  # discard the rest
    else:
        render_colors, render_alphas = rasterize_to_pixels(
            means2d,
            conics,
            colors,
            opacities,
            betas,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
            packed=packed,
            absgrad=absgrad,
        )
    if render_mode in ["ED", "RGB+ED"]:
        # normalize the accumulated depth to get the expected depth
        render_colors = torch.cat(
            [
                render_colors[..., :-1],
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ],
            dim=-1,
        )
    if render_mode == "Normal":
        render_colors = depth_to_normal(render_colors, camtoworlds, Ks)
        render_colors = (render_colors + 1) / 2

    return render_colors, render_alphas, meta


def _rasterization(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    betas: Tensor,  # [N]
    colors: Tensor,  # [(C,) N, K, 3] — DC SH coefficients
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    eps2d: float = 0.3,
    sb_number: Optional[int] = None,
    sb_params: Optional[Tensor] = None,  # [N, sb_number, 6]
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal[
        "RGB", "Normal", "Diffuse", "Specular", "Depth", "EDepth", "RGB+D", "RGB+ED"
    ] = "RGB",
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    batch_per_iter: int = 100,
) -> Tuple[Tensor, Tensor, Dict]:
    """A version of rasterization() that utilies on PyTorch's autograd.

    .. note::
        This function still relies on gsplat's CUDA backend for some computation, but the
        entire differentiable graph is on of PyTorch (and nerfacc) so could use Pytorch's
        autograd for backpropagation.

    .. note::
        This function relies on installing latest nerfacc, via:
        pip install git+https://github.com/nerfstudio-project/nerfacc

    .. note::
        Compared to rasterization(), this function does not support some arguments such as
        `packed`, `sparse_grad` and `absgrad`.
    """
    from gsplat.cuda._torch_impl import (
        _fully_fused_projection,
        _quat_scale_to_covar_preci,
        _rasterize_to_pixels,
    )

    N = means.shape[0]
    C = viewmats.shape[0]
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert opacities.shape == (N,), opacities.shape
    assert betas.shape == (N,), betas.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape
    assert render_mode in [
        "RGB",
        "Normal",
        "Diffuse",
        "Specular",
        "Depth",
        "EDepth",
        "RGB+D",
        "RGB+ED",
    ], render_mode

    # colors: [N, K, 3] or [C, N, K, 3] (DC SH coefficients, K >= 1)
    assert (colors.dim() == 3 and colors.shape[0] == N and colors.shape[2] == 3) or (
        colors.dim() == 4 and colors.shape[:2] == (C, N) and colors.shape[3] == 3
    ), colors.shape
    if sb_number:
        assert sb_params.shape == (C, N, sb_number, 6) or sb_params.shape == (
            N,
            sb_number,
            6,
        ), sb_params.shape

    # Project Gaussians to 2D.
    # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
    covars, _ = _quat_scale_to_covar_preci(quats, scales, True, False, triu=False)
    radii, means2d, depths, conics, compensations = _fully_fused_projection(
        means,
        covars,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        near_plane=near_plane,
        far_plane=far_plane,
        calc_compensations=(rasterize_mode == "antialiased"),
    )
    opacities = opacities.repeat(C, 1)  # [C, N]
    betas = betas.repeat(C, 1)  # [C, N]
    camera_ids, primitive_ids = None, None

    if compensations is not None:
        opacities = opacities * compensations

    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=False,
        n_cameras=C,
        camera_ids=camera_ids,
        primitive_ids=primitive_ids,
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    # Evaluate degree-0 SH to get base RGB, then apply spherical beta.
    SH_C0 = 0.28209479177387814
    camtoworlds = torch.inverse(viewmats)  # [C, 4, 4]
    dirs = means[None, :, :] - camtoworlds[:, None, :3, 3]  # [C, N, 3]
    masks = radii > 0  # [C, N]
    shs = colors.expand(C, -1, -1, -1) if colors.dim() == 3 else colors  # [C, N, K, 3]
    colors = torch.clamp_min(shs[..., 0, :] * SH_C0 + 0.5, 0.0)
    if sb_number:
        if sb_params.dim() == 3:
            sb_params = sb_params.expand(C, -1, -1, -1)
        colors = spherical_beta(sb_number, dirs, colors, sb_params, masks=masks)
        colors = torch.clamp_min(colors, 0.0)

    # Rasterize to pixels
    if render_mode in ["RGB+D", "RGB+ED"]:
        colors = torch.cat((colors, depths[..., None]), dim=-1)
        if backgrounds is not None:
            backgrounds = torch.cat(
                [backgrounds, torch.zeros(C, 1, device=backgrounds.device)], dim=-1
            )
    elif render_mode in ["Depth", "EDepth"]:
        colors = depths[..., None]
        if backgrounds is not None:
            backgrounds = torch.zeros(C, 1, device=backgrounds.device)
    else:  # RGB
        pass
    if colors.shape[-1] > channel_chunk:
        # slice into chunks
        n_chunks = (colors.shape[-1] + channel_chunk - 1) // channel_chunk
        render_colors, render_alphas = [], []
        for i in range(n_chunks):
            colors_chunk = colors[..., i * channel_chunk : (i + 1) * channel_chunk]
            backgrounds_chunk = (
                backgrounds[..., i * channel_chunk : (i + 1) * channel_chunk]
                if backgrounds is not None
                else None
            )
            render_colors_, render_alphas_ = _rasterize_to_pixels(
                means2d,
                conics,
                colors_chunk,
                opacities,
                betas,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds_chunk,
                batch_per_iter=batch_per_iter,
            )
            render_colors.append(render_colors_)
            render_alphas.append(render_alphas_)
        render_colors = torch.cat(render_colors, dim=-1)
        render_alphas = render_alphas[0]  # discard the rest
    else:
        render_colors, render_alphas = _rasterize_to_pixels(
            means2d,
            conics,
            colors,
            opacities,
            betas,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
            batch_per_iter=batch_per_iter,
        )
    if render_mode in ["ED", "RGB+ED"]:
        # normalize the accumulated depth to get the expected depth
        render_colors = torch.cat(
            [
                render_colors[..., :-1],
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ],
            dim=-1,
        )

    meta = {
        "camera_ids": camera_ids,
        "primitive_ids": primitive_ids,
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "conics": conics,
        "opacities": opacities,
        "betas": betas,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "n_cameras": C,
    }
    return render_colors, render_alphas, meta
