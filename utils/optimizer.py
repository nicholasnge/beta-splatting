import numpy as np
import torch
from torch import nn


def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    """Log-linear LR decay from lr_init to lr_final over max_steps."""
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp
    return helper


def cat_tensors_to_optimizer(optimizer, tensors_dict):
    """Extend optimizer state when new Gaussians are added."""
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        assert len(group["params"]) == 1
        extension_tensor = tensors_dict[group["name"]]
        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat(
                (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
            )
            stored_state["exp_avg_sq"] = torch.cat(
                (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
            )
            del optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(
                torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
            )
            optimizer.state[group["params"][0]] = stored_state
        else:
            group["params"][0] = nn.Parameter(
                torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
            )
        optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors
