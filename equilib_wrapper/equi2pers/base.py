from typing import Union, Tuple
import numpy as np
import torch
from equilib_wrapper import camera
from equilib_wrapper.equi2pers.runner_numpy import run as run_numpy
from equilib_wrapper.equi2pers.runner_torch import run as run_torch

ArrayLike = Union[np.ndarray, torch.Tensor]

def equi2pers(
    equi: ArrayLike,
    rotation: ArrayLike,
    height: int,
    width: int,
    hfov: float = None,
    vfov: float = None,
    skew: float = 0.0,
    hrange: Tuple[float, float] = None,
    vrange: Tuple[float, float] = None,
    degrees: bool = False,
    fill_mode: str = "wrap",
    pad_value: ArrayLike = (0, 0, 0),
    mode: str = "bilinear",
    z_down: bool = False,
    clip_output: bool = True,
    **kwargs,
) -> ArrayLike:
    """Extract perspective images from equirectangular images.

    Args:
        equi (ArrayLike): Equirectangular image with shape (C, H, W) or (B, C, H, W).
        rotation (ArrayLike): Rotation angles in extrinsic XYZ (roll, pitch, yaw) order. See 'degrees' for unit.
        height (int): Height of the perspective image.
        width (int): Width of the perspective image.
        hfov (float, optional): Horizontal field of view in degrees. Only one of hfov and vfov can be specified. Defaults to None.
        vfov (float, optional): Vertical field of view in degrees. Only one of hfov and vfov can be specified. Defaults to None.
        hrange (Tuple[float, float], optional): Horizontal range of the equirectangular image in radians. See 'degrees' for unit. If None, the full range is used. Defaults to (-np.pi, np.pi).
        vrange (Tuple[float, float], optional): Vertical range of the equirectangular image in radians. See 'degrees' for unit. If None, the full range is used. Defaults to (-np.pi / 2, np.pi / 2).
        degrees (bool, optional): Whether angles are in degrees. Defaults to False.
        fill_mode (str, optional): Fill mode for regions outside of the original equirectangular image. Defaults to "wrap".
        pad_value (ArrayLike, optional): Fill value for the pixels that are outside of the original equirectangular image. Only used when fill_mode is "pad". Defaults to (0, 0, 0).
        mode (str, optional): Sampling mode for grid_sample. Defaults to "bilinear".
        z_down (bool, optional): Whether the z-axis is pointing down. Defaults to False.
        clip_output (bool, optional): Whether to clip the output to the range of the input equirectangular image. Defaults to True.

    Returns:
        ArrayLike: Perspective image with shape (C, height, width) or (B, C, height, width).
    """
    
    # Check input validity
    if hfov and vfov:
        raise ValueError("Only one of hfov and vfov can be specified.")
    # if (hrange[0] > hrange[1] or vrange[0] > vrange[1]
    #     or hrange[0] < -np.pi or hrange[1] > np.pi
    #     or vrange[0] < -np.pi / 2 or vrange[1] > np.pi / 2): # update these checks to match 'degrees'
    #     raise ValueError("Invalid hrange or vrange.")
    if fill_mode not in ["wrap", "pad"]:
        raise ValueError("Invalid fill_mode.")
    
    if degrees:
        rotation = np.radians(rotation)
        if hfov:
            hfov = np.radians(hfov)
        if vfov:
            vfov = np.radians(vfov)
        
        hrange = (-np.pi, np.pi) if hrange is None else tuple(np.radians(h) for h in hrange)
        vrange = (-np.pi / 2, np.pi / 2) if vrange is None else tuple(np.radians(v) for v in vrange)
    else:
        hrange = (-np.pi, np.pi) if hrange is None else hrange
        vrange = (-np.pi / 2, np.pi / 2) if vrange is None else vrange
    
    if vfov:
        hfov = camera.vfov_to_hfov(vfov, width, height)
    
    _type = None
    if isinstance(equi, np.ndarray):
        _type = "numpy"
    elif isinstance(equi, torch.Tensor):
        _type = "torch"
    
    batched = equi.ndim == 4

    if _type == "numpy":
        if not batched:
            equi = equi[np.newaxis]

        out = run_numpy(
            equi=equi,
            rots=[dict(zip(("roll", "pitch", "yaw"), rotation))],
            height=height,
            width=width,
            fov_x=np.rad2deg(hfov),
            skew=skew,
            z_down=z_down,
            mode=mode,
            h_range=hrange,
            v_range=vrange,
            clip_output=clip_output,
            fill_value=None if fill_mode == "wrap" else pad_value,
            **kwargs
            )
    elif _type == "torch":
        if not batched:
            equi = equi.unsqueeze(0)
        
        out = run_torch(
            equi=equi,
            rots=[dict(zip(("roll", "pitch", "yaw"), rotation))],
            height=height,
            width=width,
            fov_x=np.rad2deg(hfov),
            skew=skew,
            z_down=z_down,
            mode=mode,
            h_range=hrange,
            v_range=vrange,
            clip_output=clip_output,
            fill_value=None if fill_mode == "wrap" else pad_value,
            **kwargs
            )
    else:
        raise ValueError("Input type is not supported.")

    if not batched:
        out = out[0]

    return out