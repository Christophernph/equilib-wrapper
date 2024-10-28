from typing import Dict, List, Tuple

import torch
from equilib.torch_utils import (create_rotation_matrices, get_device)
from equilib.equi2pers.torch import prep_matrices, matmul, convert_grid
from equilib_wrapper.equi2pers.grid_sample import grid_sample

def run(
    equi: torch.Tensor,
    rots: List[Dict[str, float]],
    height: int,
    width: int,
    fov_x: float,
    skew: float,
    z_down: bool,
    mode: str,
    h_range: Tuple[float, float],
    v_range: Tuple[float, float],
    clip_output: bool = True,
    fill_value: Tuple[float, float, float] = None,
    backend: str = "native",
) -> torch.Tensor:
    
    assert equi.ndim == 4, f"Input should be 4-dimensional (B, C, H, W), but got {equi.ndim}."
    assert len(equi) == len(rots), f"Batch size of equi and rot differs: {len(equi)} vs {len(rots)}."
    
    if equi.shape[0] > 1:
        raise NotImplementedError("Batched input larger than one is not supported yet.")

    equi_dtype = equi.dtype
    assert equi_dtype in (
        torch.uint8,
        torch.float16,
        torch.float32,
        torch.float64,
    ), f"Equirectangular image has dtype of {equi_dtype}, which is incompatible. Try {torch.uint8}, {torch.float16}, {torch.float32}, or {torch.float64}." 

    # Convert to float32 if uint8, else keep the dtype
    if equi.device.type == "cuda":
        dtype = torch.float32 if equi_dtype == torch.uint8 else equi_dtype
    else:
        # NOTE: for cpu, it can't use half-precision
        dtype = torch.float32 if equi_dtype == torch.uint8 else equi_dtype
        assert dtype != torch.float16, f"Equirectangular image has dtype of {equi_dtype}, which is not supported on cpu. Try {torch.uint8}, {torch.float32}, or {torch.float64}."

    if backend == "native" and equi_dtype == torch.uint8:
        # FIXME: hacky way of dealing with images that are uint8 when using
        # torch.grid_sample
        equi = equi.type(torch.float32)

    bs, c, h_equi, w_equi = equi.shape

    # Update shape based on h_range and v_range
    h_equi = int((torch.pi / (v_range[1] - v_range[0])) * h_equi)
    w_equi = int((2 * torch.pi / (h_range[1] - h_range[0])) * w_equi)

    img_device = get_device(equi)

    # initialize output tensor
    if backend == "native":
        # NOTE: don't need to initialize for `native`
        out = None
    else:
        out = torch.empty(
            (bs, c, height, width), dtype=dtype, device=img_device
        )

    # FIXME: for now, calculate the grid in cpu
    # I need to benchmark performance of it when grid is created on cuda
    tmp_device = torch.device("cpu")
    if equi.device.type == "cuda" and dtype == torch.float16:
        tmp_dtype = torch.float32
    else:
        tmp_dtype = dtype

    # create grid and transform matrix
    m, G = prep_matrices(
        height=height,
        width=width,
        batch=bs,
        fov_x=fov_x,
        skew=skew,
        dtype=tmp_dtype,
        device=tmp_device,
    )

    # create batched rotation matrices
    R = create_rotation_matrices(
        rots=rots, z_down=z_down, dtype=tmp_dtype, device=tmp_device
    )

    # rotate and transform the grid
    M = matmul(m, G, R)

    # create a pixel map grid
    grid = convert_grid(M=M, h_equi=h_equi, w_equi=w_equi, method="robust")
    grid = grid.to(img_device, dtype=equi.dtype)

    # Correct grid according to h_range and v_range
    x_shift = (h_range[0] + torch.pi) / (2 * torch.pi) * w_equi
    y_shift = (v_range[0] + torch.pi / 2) / torch.pi * h_equi
    grid = grid - torch.tensor([y_shift, x_shift]).reshape(1, 2, 1, 1)

    if fill_value is not None:
        # grid_sample modifies the grid, so we have to compute indices before
        # We want to zero out the pixels that are outside of the original equi image
        # That means places where 
        #   grid[:, 0, ...] (y) < 0
        #   grid[:, 1, ...] (x) < 0
        #   grid[:, 0, ...] (y) > h_equi
        #   grid[:, 1, ...] (x) > w_equi
        h_equi, w_equi = equi.shape[-2:]
        indices = torch.nonzero((grid[:, 0, ...] < 0) | (grid[:, 1, ...] < 0) | (grid[:, 0, ...] > h_equi) | (grid[:, 1, ...] > w_equi))

    # grid sample
    out = grid_sample(
        img=equi,
        grid=grid,
        out=out,  # FIXME: is this necessary?
        mode=mode,
        backend=backend,
    )

    # NOTE: we assume that `out` keeps it's dtype

    out = (
        out.type(equi_dtype)
        if equi_dtype == torch.uint8 or not clip_output
        else torch.clip(out, torch.min(equi), torch.max(equi))
    )

    if fill_value is not None:
        out[indices[..., 0], :, indices[..., 1], indices[..., 2]] = torch.tensor(fill_value).to(out.device, dtype=out.dtype)

    return out