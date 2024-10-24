from typing import Union, Tuple
import numpy as np
import torch
from equilib_wrapper import camera
from equilib_wrapper.equi2pers.runner import run

ArrayLike = Union[np.ndarray, torch.Tensor]

def equi2pers(
    equi: ArrayLike,
    rotation: ArrayLike,
    height: int,
    width: int,
    hfov: float = None,
    vfov: float = None,
    h_range: Tuple[float, float] = (-np.pi, np.pi),
    v_range: Tuple[float, float] = (-np.pi / 2, np.pi / 2),
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
        h_range (Tuple[float, float], optional): Horizontal range of the equirectangular image in radians. See 'degrees' for unit. Defaults to (-np.pi, np.pi).
        v_range (Tuple[float, float], optional): Vertical range of the equirectangular image in radians. See 'degrees' for unit. Defaults to (-np.pi / 2, np.pi / 2).
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
    if (h_range[0] > h_range[1] or v_range[0] > v_range[1]
        or h_range[0] < -np.pi or h_range[1] > np.pi
        or v_range[0] < -np.pi / 2 or v_range[1] > np.pi / 2):
        raise ValueError("Invalid h_range or v_range.")
    if fill_mode not in ["wrap", "pad"]:
        raise ValueError("Invalid fill_mode.")
    
    if degrees:
        rotation = np.radians(rotation)
        if hfov:
            hfov = np.radians(hfov)
        if vfov:
            vfov = np.radians(vfov)
        h_range = tuple(np.radians(h) for h in h_range)
        v_range = tuple(np.radians(v) for v in v_range)
    
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

        out = run(
                equi=equi,
                rots=[dict(zip(("roll", "pitch", "yaw"), rotation))],
                height=height,
                width=width,
                fov_x=np.rad2deg(hfov),
                z_down=z_down,
                mode=mode,
                h_range=h_range,
                v_range=v_range,
                clip_output=clip_output,
                fill_value=None if fill_mode == "reflect" else pad_value,
                **kwargs
            )
    elif _type == "torch":
        raise NotImplementedError("Torch support is not implemented yet.")
    else:
        raise ValueError("Input type is not supported.")

    if not batched:
        out = out[0]

    return out

if __name__ == "__main__":
    from PIL import Image

    equi = Image.open("50.png").convert("RGB")
    equi = np.asarray(equi).transpose(2, 0, 1)
    equi = equi[..., equi.shape[-1] // 2:]

    rotation = np.array([20, 0, 0])
    height = 1024
    width = 1536
    hfov = 90
    result = equi2pers(equi,
                       rotation,
                       height,
                       width,
                       hfov=hfov,
                       degrees=True,
                       h_range=(0, np.pi)
                       )
    
    result = result.transpose(1, 2, 0)
    result = Image.fromarray(result)
    result.save("result.png")