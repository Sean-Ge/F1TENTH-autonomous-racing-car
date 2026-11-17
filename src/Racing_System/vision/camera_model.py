"""
Pinhole camera model for the RealSense colour camera.

Everything the vision follower knows about distance comes from here. We assume the
floor is flat and the camera is a simple pinhole at a known height and tilt, so every
image row below the horizon lines up with one distance on the floor. That lets us turn
pixels into metres without using the depth stream at all.

Image coordinates: u = column (0 at the left), v = row (0 at the top).
Floor coordinates: forward_m ahead of the camera, left_m to the left of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """
    Pinhole intrinsics of the colour camera, all in pixels.

    Attributes:
        fx (float): Focal length along the image columns.
        fy (float): Focal length along the image rows.
        cx (float): Column of the optical centre.
        cy (float): Row of the optical centre.
        width (int): Image width.
        height (int): Image height.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class CameraPose:
    """
    Where the camera sits on the car.

    Attributes:
        height_m (float): Height of the lens above the floor in metres.
        pitch_rad (float): Tilt of the camera in radians, positive = tilted up.
    """

    height_m: float
    pitch_rad: float


# typical RealSense D435 colour intrinsics at 640x480
COLOR_DEFAULT = Intrinsics(615.0, 615.0, 320.0, 240.0, 640, 480)
# our original mount (tilted up 13.9 deg). vision_follow_node now builds its pose from
# camera_height_m and camera_pitch_deg in params.yaml instead of using this
POSE_DEFAULT = CameraPose(0.19, math.radians(13.9))


def _floor_depth(dy: np.ndarray, pose: CameraPose) -> np.ndarray:
    """
    Find how far along the optical axis a ray travels before it hits the floor.

    Args:
        dy (np.ndarray): Vertical slope of each ray, (v - cy) / fy.
        pose (CameraPose): Camera height and pitch.

    Returns:
        np.ndarray: Optical depth (m) where each ray meets the floor, or nan if it never does.
    """
    # how quickly the ray drops toward the floor once you account for the camera tilt
    den = dy * math.cos(pose.pitch_rad) - math.sin(pose.pitch_rad)
    z = np.full(den.shape, np.nan)
    
    # rays at or above the horizon never come down to the floor, so they stay nan
    hits = den > 0.0
    z[hits] = pose.height_m / den[hits]
    return z


def horizon_row(intr: Intrinsics, pose: CameraPose) -> float:
    """
    Find the image row where the floor would meet the horizon.

    Args:
        intr (Intrinsics): Camera intrinsics.
        pose (CameraPose): Camera height and pitch.

    Returns:
        float: Row of the horizon. Only rows below it can see the floor.
    """
    return intr.cy + intr.fy * math.tan(pose.pitch_rad)


def floor_depth_per_row(intr: Intrinsics, pose: CameraPose) -> np.ndarray:
    """
    Work out the floor's optical depth for every row of the image.

    On a flat floor the depth only depends on the row, not the column, so one
    value per row is enough.

    Args:
        intr (Intrinsics): Camera intrinsics.
        pose (CameraPose): Camera height and pitch.

    Returns:
        np.ndarray: One depth (m) per image row, nan for rows at or above the horizon.
    """
    rows = np.arange(intr.height, dtype=np.float64)
    return _floor_depth((rows - intr.cy) / intr.fy, pose)


def pixel_to_floor(u: np.ndarray, v: np.ndarray, intr: Intrinsics,
                   pose: CameraPose) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project image pixels down onto the floor.

    Args:
        u (np.ndarray): Pixel columns.
        v (np.ndarray): Pixel rows (same shape as u).
        intr (Intrinsics): Camera intrinsics.
        pose (CameraPose): Camera height and pitch.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (forward_m, left_m) of each pixel's floor point,
        nan where the pixel doesn't see the floor.
    """
    cos_t = math.cos(pose.pitch_rad)
    sin_t = math.sin(pose.pitch_rad)
    # normalised ray direction through each pixel
    dx = (np.asarray(u, dtype=np.float64) - intr.cx) / intr.fx
    dy = (np.asarray(v, dtype=np.float64) - intr.cy) / intr.fy
    z = _floor_depth(dy, pose)
    # rotate the optical depth back into a flat forward distance
    forward = z * (cos_t + dy * sin_t)
    # columns right of centre are to the car's right, so left is negative there
    left = -z * dx
    return forward, left


def first_in_range_row(intr: Intrinsics, pose: CameraPose, max_range_m: float) -> int:
    """
    Find the highest image row whose floor point is no further than max_range_m.

    Rows above this one are ignored by the follower since they look too far ahead.

    Args:
        intr (Intrinsics): Camera intrinsics.
        pose (CameraPose): Camera height and pitch.
        max_range_m (float): Furthest floor distance we care about (m).

    Returns:
        int: Index of that row.

    Raises:
        ValueError: If no row sees the floor within max_range_m (camera pointing too high).
    """
    rows = np.arange(intr.height, dtype=np.float64)
    # check the floor distance straight down the middle column of every row
    forward, _ = pixel_to_floor(np.full_like(rows, intr.cx), rows, intr, pose)
    in_range = forward <= max_range_m
    if not in_range.any():
        if np.isnan(forward).all():
            nearest = "none (the camera does not see the floor)"
        else:
            nearest = f"{np.nanmin(forward):.2f} m"
        raise ValueError(f"no image row sees the floor within max_range_m = {max_range_m:.2f} m; "
                         f"nearest visible floor point: {nearest}")
    # argmax gives the first True, which is the topmost row in range
    return int(np.argmax(in_range))


def push_px_per_row(intr: Intrinsics, pose: CameraPose, half_width_m: float) -> np.ndarray:
    """
    Convert the car's half-width into pixels for every image row.

    Something far away looks small, so the same 0.18 m is a lot of pixels near the
    bottom of the image and only a few near the horizon.

    Args:
        intr (Intrinsics): Camera intrinsics.
        pose (CameraPose): Camera height and pitch.
        half_width_m (float): How far to grow obstacles sideways (m).

    Returns:
        np.ndarray: Pixels to push obstacles sideways in each row (0 for rows that miss the floor).
    """
    z = floor_depth_per_row(intr, pose)
    push = np.zeros(intr.height, dtype=np.int64)
    hits = ~np.isnan(z)
    # metres to pixels at that depth, rounded up so we never under-pad
    push[hits] = np.ceil(half_width_m * intr.fx / z[hits]).astype(np.int64)
    return push
