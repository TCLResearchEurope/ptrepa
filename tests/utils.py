import torch


def mask_channels(x: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
    # TODO Reimplement this in a vector form
    x_masked = torch.zeros_like(x, device=x.device)
    for b in range(x.shape[0]):
        for c in range(x.shape[1]):
            if channel_mask[c]:
                x_masked[b, c, :, :] = x[b, c, :, :]

    return x_masked
