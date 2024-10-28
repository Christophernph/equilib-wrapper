from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from equilib.equi2pers.numpy import (convert_grid, create_rotation_matrices,
                                     matmul, numpy_grid_sample, prep_matrices)


def run(
    equi: np.ndarray,
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
) -> np.ndarray:
    """Extract perspective images from equirectangular images

    Args:
        equi (np.ndarray): Equirectangular image with shape (B, C, H, W). Acceptable dtypes are uint8, float32, and float64.
        rots (List[Dict[str, float]]): List of rotation angles in degrees. Each element is a dictionary with keys 'yaw', 'pitch', and 'roll'.
        height (int): Height of the perspective image.
        width (int): Width of the perspective image.
        fov_x (float): Field of view in degrees along the horizontal axis.
        skew (float): Skew of the camera.
        z_down (bool): Whether the z-axis is pointing down.
        mode (str): Sampling mode for grid_sample.
        h_range (Tuple[float, float]): Horizontal range of the equirectangular image in radians.
        v_range (Tuple[float, float]): Vertical range of the equirectangular image in radians.
        clip_output (bool, optional): Whether to clip the output to the range of the input equirectangular image. Defaults to True.
        fill_value (Tuple[float, float, float], optional): Fill value for the pixels that are outside of the original equirectangular image. Defaults to None. If None, regions outside of the original image wrap around.

    Returns:
        np.ndarray: Perspective image with shape (B, C, height, width).
    """

    assert equi.ndim == 4, f"Input should be 4-dimensional (B, C, H, W), but got {equi.ndim}."
    assert len(equi) == len(rots), f"Batch size of equi and rot differs: {len(equi)} vs {len(rots)}."
    
    if equi.shape[0] > 1:
        raise NotImplementedError("Batched input larger than one is not supported yet.")

    equi_dtype = equi.dtype
    assert equi_dtype in (np.uint8, np.float32, np.float64), f"Equirectangular image has dtype of {equi_dtype}, which is incompatible. Try {np.uint8}, {np.float32}, or {np.float64}."

    # Convert to float32 if uint8, else keep the dtype
    dtype = np.dtype(np.float32) if equi_dtype == np.dtype(np.uint8) else equi_dtype

    bs, c, h_equi, w_equi = equi.shape

    # Update shape based on h_range and v_range
    h_equi = int((np.pi / (v_range[1] - v_range[0])) * h_equi)
    w_equi = int((2 * np.pi / (h_range[1] - h_range[0])) * w_equi)

    # initialize output array
    out = np.empty((bs, c, height, width), dtype=dtype)

    # create grid and transfrom matrix
    m, G = prep_matrices(
        height=height,
        width=width,
        batch=bs,
        fov_x=fov_x,
        skew=skew,
        dtype=dtype,
    )

    # create batched rotation matrices
    R = create_rotation_matrices(rots=rots, z_down=z_down, dtype=dtype)

    # rotate and transform the grid
    M = matmul(m, G, R)

    # create a pixel map grid
    grid = convert_grid(M=M, h_equi=h_equi, w_equi=w_equi, method="robust") # yx
    
    # Correct grid according to h_range and v_range
    x_shift = (h_range[0] + np.pi) / (2 * np.pi) * w_equi
    y_shift = (v_range[0] + np.pi / 2) / np.pi * h_equi
    grid = grid - np.array([y_shift, x_shift]).reshape(1, 2, 1, 1)

    # grid sample
    out = numpy_grid_sample(
        img=equi,
        grid=grid,
        out=out,  # FIXME: pass-by-reference confusing?
        mode=mode,
    )

    out = (
        out.astype(equi_dtype)
        if equi_dtype == np.dtype(np.uint8) or not clip_output
        else np.clip(out, np.min(equi), np.max(equi))
    )

    if fill_value is not None:
        # We want to zero out the pixels that are outside of the original equi image
        # That means places where 
        #   grid[:, 0, ...] (y) < 0
        #   grid[:, 1, ...] (x) < 0
        #   grid[:, 0, ...] (y) > h_equi
        #   grid[:, 1, ...] (x) > w_equi
        h_equi, w_equi = equi.shape[-2:]
        indices = np.nonzero((grid[:, 0, ...] < 0) | (grid[:, 1, ...] < 0) | (grid[:, 0, ...] > h_equi) | (grid[:, 1, ...] > w_equi))
        out[indices[0], :, indices[1], indices[2]] = fill_value
    return out