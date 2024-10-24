import numpy as np
from numpy.typing import NDArray


def fov_to_focal(fov: float, size: int) -> float:
    """
    Convert FOV to focal size.

    Parameters
    ----------
    fov : float
        FOV in radians.
    size : int
        Size of FOV edge in pixels.

    Returns
    -------
    float
        Focal size in pixels.
    """
    A = size / 2
    a = fov / 2
    return A / np.tan(a)


def focal_to_fov(f: float, size: int) -> float:
    """
    Convert focal size in pixels to FOV in radians.

    Parameters
    ----------
    f : float
        Focal size in pixels.
    size : int
        Size of edge in pixels.

    Returns
    -------
    float
        FOV in radians.
    """
    return 2 * np.arctan(size / (2 * f))


def dfov_to_hfov(dfov: float, width: int, height: int):
    """Convert diagonal FOV to HFOV."""
    diag = np.sqrt(width**2 + height**2)
    f = fov_to_focal(dfov, diag)
    return focal_to_fov(f, width)


def vfov_to_hfov(vfov: float, width: int, height: int):
    """Convert vertical FOV to horizontal FOV."""
    fov = fov_to_focal(vfov, height)
    return focal_to_fov(fov, width)


def hfov_to_vfov(hfov: float, width: int, height: int):
    """Convert horizontal FOV to vertical FOV."""
    fov = fov_to_focal(hfov, width)
    return focal_to_fov(fov, height)
