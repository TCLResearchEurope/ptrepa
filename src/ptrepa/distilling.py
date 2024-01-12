import copy
import logging

import torch

__all__ = ["distill_batchnorm2d_in_place", "distill_conv2d_in_place"]


logger = logging.getLogger(__name__)


def distill_batchnorm2d_in_place(
    m: torch.nn.BatchNorm2d, predecessor_mask: torch.Tensor, output_mask: torch.Tensor
) -> None:
    torch.testing.assert_close(predecessor_mask, output_mask)

    if m.weight is not None:
        weight_new = copy.deepcopy(m.weight.data[predecessor_mask])
        m.weight.data = weight_new.to(m.weight.data.device)
    if m.bias is not None:
        bias_new = copy.deepcopy(m.bias.data[predecessor_mask])
        m.bias.data = bias_new.to(m.bias.data.device)
    assert isinstance(m.running_mean, torch.Tensor)
    running_mean_new = copy.deepcopy(m.running_mean[predecessor_mask])
    m.running_mean = running_mean_new.to(m.running_mean.device)
    assert isinstance(m.running_var, torch.Tensor)
    running_var_new = copy.deepcopy(m.running_var[predecessor_mask])
    m.running_var = running_var_new.to(m.running_var.device)
    num_features = int(torch.sum(predecessor_mask).item())
    m.num_features = num_features


def distill_conv2d_in_place(
    m: torch.nn.Conv2d, predecessor_mask: torch.Tensor, output_mask: torch.Tensor
) -> None:
    if m.groups == 1:
        logger.info(f"{m.weight.data.shape=}")
        logger.info(f"{m.weight.shape=}")
        # Fix weight
        weight_new = copy.deepcopy(m.weight.data[output_mask, :, :, :])
        weight_new = weight_new[:, predecessor_mask, :, :]
        m.weight.data = weight_new.to(m.weight.data.device)
        logger.info(f"{m.weight.data.shape=}")
        logger.info(f"{m.weight.shape=}")
        # Fix bias
        if hasattr(m, "bias") and m.bias is not None:
            bias_new = copy.deepcopy(m.bias.data[output_mask])
            m.bias.data = bias_new.to(m.bias.data.device)

        # Fix parameters
        in_channels = int(torch.sum(predecessor_mask).item())
        out_channels = int(torch.sum(output_mask).item())
        m.in_channels = in_channels
        m.out_channels = out_channels

    elif m.groups == m.in_channels and m.in_channels == m.out_channels:
        if m.in_channels != m.out_channels:
            msg = "Depthwise convolutions with expansion ration >1 not supported"
            raise ValueError(msg)

        # Fix weight
        weight_new = copy.deepcopy(m.weight.data[predecessor_mask, :, :, :])
        m.weight.data = weight_new.to(m.weight.data.device)

        # Fix bias
        if hasattr(m, "bias") and m.bias is not None:
            bias_new = copy.deepcopy(m.bias.data[output_mask])
            m.bias.data = bias_new.to(m.bias.data.device)

        # Fix parameters
        in_channels = int(torch.sum(predecessor_mask).item())
        m.in_channels = in_channels
        m.out_channels = in_channels
        m.groups = in_channels
    else:
        raise NotImplementedError(
            "Distilling conv2d for "
            f"{m.groups=}, {m.in_channels=}, {m.out_channels} not implemented"
        )
