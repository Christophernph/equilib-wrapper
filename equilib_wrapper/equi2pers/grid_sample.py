import warnings
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F

DTYPES = (torch.uint8, torch.float16, torch.float32, torch.float64)


def grid_sample(
    img: torch.Tensor,
    grid: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    mode: str = "bilinear",
    backend: str = "native",
) -> torch.Tensor:
    """Torch grid sampling algorithm

    params:
    - img (torch.Tensor)
    - grid (torch.Tensor)
    - out (Optional[torch.Tensor]): defaults to None
    - mode (str): ('bilinear', 'bicubic', 'nearest')
    - backend (str): ('native', 'pure')

    return:
    - img (torch.Tensor)

    NOTE: for `backend`, `pure` is relatively efficient since grid doesn't need
    to be in the same device as the `img`. However, `native` is faster.

    NOTE: for `pure` backends, we need to pass reference to `out`.

    NOTE: for `native` backends, we should pass anything for `out`

    """

    if backend == "native":
        if out is not None:
            # NOTE: out is created
            warnings.warn(
                "don't need to pass preallocated `out` to `grid_sample`"
            )
        assert img.device == grid.device, (
            f"ERR: when using {backend}, the devices of `img` and `grid` need"
            "to be on the same device"
        )
        if mode == "nearest":
            out = native_nearest(img, grid)
        elif mode == "bilinear":
            out = native_bilinear(img, grid)
        elif mode == "bicubic":
            out = native_bicubic(img, grid)
        else:
            raise ValueError(f"ERR: {mode} is not supported")
    else:
        raise NotImplementedError("Only `native` backend is supported")

    return out

def native(
    img: torch.Tensor, grid: torch.Tensor, mode: str = "bilinear"
) -> torch.Tensor:
    """Torch Grid Sample (default)

    - Uses `torch.nn.functional.grid_sample`
    - By far the best way to sample

    params:
    - img (torch.Tensor): Tensor[B, C, H, W]  or Tensor[C, H, W]
    - grid (torch.Tensor): Tensor[B, 2, H, W] or Tensor[2, H, W]
    - device (int or str): torch.device
    - mode (str): (`bilinear`, `bicubic`, `nearest`)

    returns:
    - out (torch.Tensor): Tensor[B, C, H, W] or Tensor[C, H, W]
        where H, W are grid size

    NOTE: `img` and `grid` needs to be on the same device

    NOTE: `img` and `grid` is somehow mutated (inplace?), so if you need
    to reuse `img` and `grid` somewhere else, use `.clone()` before
    passing it to this function

    NOTE: this method is different from other grid sampling that
    the padding cannot be wrapped. There might be pixel inaccuracies
    when sampling from the boundaries of the image (the seam).

    I hope later on, we can add wrap padding to this since the function
    is super fast.

    """

    assert (
        grid.dtype == img.dtype
    ), "ERR: img and grid should have the same dtype"

    _, _, h, w = img.shape

    # grid in shape: (batch, channel, h_out, w_out)
    # grid out shape: (batch, h_out, w_out, channel)
    grid = grid.permute(0, 2, 3, 1)

    """Preprocess for grid_sample
    normalize grid -1 ~ 1

    assumptions:
    - values of `grid` is between `0 ~ (h-1)` and `0 ~ (w-1)`
    - input of `grid_sample` need to be between `-1 ~ 1`
    - maybe lose some precision when we map the values (int to float)?

    mapping (e.g. mapping of height):
    1. 0 <= y <= (h-1)
    2. -1/2 <= y' <= 1/2  <- y' = y/(h-1) - 1/2
    3. -1 <= y" <= 1  <- y" = 2y'
    """

    # FIXME: this is not necessary when we are already preprocessing grid before
    # this method is called
    grid[..., 0] %= h
    grid[..., 1] %= w

    norm_uj = torch.clamp(2 * grid[..., 0] / (h - 1) - 1, -1, 1)
    norm_ui = torch.clamp(2 * grid[..., 1] / (w - 1) - 1, -1, 1)

    # reverse: grid sample takes xy, not (height, width)
    grid[..., 0] = norm_ui
    grid[..., 1] = norm_uj

    out = F.grid_sample(
        img,
        grid,
        mode=mode,
        # use center of pixel instead of corner
        align_corners=True,
        # padding mode defaults to 'zeros' and there is no 'wrapping' mode
        padding_mode="reflection",
    )

    return out


# aliases
native_nearest = partial(native, mode="nearest")
native_bilinear = partial(native, mode="bilinear")
native_bicubic = partial(native, mode="bicubic")
