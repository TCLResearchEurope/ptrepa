import logging
from typing import Any, Optional

import pytest
import torch
import utils

import ptrepa

logger = logging.getLogger(__name__)


def _make_params_repvgg2d_basic(kernel_sizes: list[int]) -> list[dict[str, Any]]:
    res = []
    templates: list[dict[str, Any]] = [
        {"in_channels": 3, "out_channels": 6, "groups": 1},
        {"in_channels": 3, "out_channels": 3, "groups": 1},
        {"in_channels": 3, "out_channels": 3, "groups": 3},
        {"in_channels": 16, "out_channels": 24, "groups": 1},
        {"in_channels": 16, "out_channels": 16, "groups": 1},
        {"in_channels": 16, "out_channels": 16, "groups": 16},
    ]
    for ks in kernel_sizes:
        for bias in [False, True]:
            for t in templates:
                res.append({**t, **{"kernel_size": ks, "bias": bias}})
    return res


TEST_PARAMS_CONV2D = _make_params_repvgg2d_basic([1, 3])


def check_distill_conv2d_in_place(
    params: dict[str, Any],
    device: torch.device,
    float64: bool = False,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
) -> None:
    rng = torch.Generator().manual_seed(12345)
    for _ in range(5):
        in_channels = params["in_channels"]
        out_channels = params["out_channels"]
        kernel_size = params["kernel_size"]
        groups = params["groups"]
        bias = params["bias"]
        predecessor_mask = torch.rand(size=(in_channels,), generator=rng) > 0.5

        # Random predecessor mask + ensure at least one  channel is left
        i = int((torch.rand(size=(1,), generator=rng) * in_channels).item())
        predecessor_mask[i] = True

        if groups == 1:
            # Random output mask + ensure at least one channel is left
            output_mask = torch.rand(size=(out_channels,), generator=rng) > 0.5
            i = int((torch.rand(size=(1,), generator=rng) * out_channels).item())
            output_mask[i] = True
        elif groups == in_channels and in_channels == out_channels:
            # For depthwise conv pruning is supported only if expansion rate == 1
            assert params["in_channels"] == params["out_channels"]
            # For depthwise conv we prune input and output in the same places
            output_mask = predecessor_mask
        else:
            raise ValueError(
                "Testing unsupported case" f"{in_channels=} {out_channels=} {groups=}"
            )
        output_mask = output_mask.to(device)
        predecessor_mask = predecessor_mask.to(device)
        in_channels_new = predecessor_mask.sum().item()
        msg = f"Distilling in_channels={in_channels} -> {in_channels_new}"
        logger.info(msg)
        out_channels_new = output_mask.sum().item()
        msg = f"Distilling out_channels={out_channels} -> {out_channels_new}"
        logger.info(msg)

        x = torch.rand(size=(16, in_channels, 32, 32), generator=rng)
        if float64:
            x = x.to(dtype=torch.float64)
        x = x.to(device)

        x_masked = utils.mask_channels(x, predecessor_mask)
        x_masked = x_masked.to(device)
        m = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups,
            kernel_size=kernel_size,
            bias=bias,
        )
        m.to(device)
        if float64:
            m.to(dtype=torch.float64)
        m.eval()

        with torch.no_grad():
            y_masked = m(x_masked)[:, output_mask, :, :]
        ptrepa.distill_conv2d_in_place(
            m, predecessor_mask=predecessor_mask, output_mask=output_mask
        )
        x_pruned = x[:, predecessor_mask, :, :]

        with torch.no_grad():
            y_pruned = m(x_pruned)

        assert y_pruned.shape[1] == out_channels_new
        assert m.weight.shape[0] == out_channels_new
        if atol is not None and rtol is not None:
            torch.testing.assert_close(y_masked, y_pruned, atol=atol, rtol=rtol)
        else:
            torch.testing.assert_close(y_masked, y_pruned)


@pytest.mark.parametrize("params", TEST_PARAMS_CONV2D)
def test_distill_conv2d_in_place(params: dict[str, Any]) -> None:
    check_distill_conv2d_in_place(params, torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@pytest.mark.parametrize("params", TEST_PARAMS_CONV2D)
def test_distill_conv2d_in_place_cuda64(params: dict[str, Any]) -> None:
    check_distill_conv2d_in_place(params, torch.device("cuda"), float64=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
@pytest.mark.parametrize("params", TEST_PARAMS_CONV2D)
def test_distill_conv2d_in_place_cuda(params: dict[str, Any]) -> None:
    check_distill_conv2d_in_place(
        params, torch.device("cuda"), atol=0.0008, rtol=1.0e-6
    )
