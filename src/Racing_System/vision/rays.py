"""
Ray casting over the blocked mask for the vision follower.

Rays fan out from the bottom-centre pixel of the image, a bit like LIDAR beams. Each
ray walks pixel by pixel until it hits something blocked, and we report how far it got
on the floor and which way it points. All the pixel positions, distances and bearings
along every ray are worked out once at startup, so each frame only needs a lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from milestone3.vision.camera_model import CameraPose, Intrinsics, pixel_to_floor


@dataclass
class RayTable:
    """
    Precomputed pixels, floor distances and bearings for every step of every ray.

    Arrays are shaped (number of rays, number of steps).

    Attributes:
        angles_rad (np.ndarray): Image-space angle of each ray, 0 = straight up the image.
        u (np.ndarray): Pixel column of each step (clipped into the image).
        v (np.ndarray): Pixel row of each step (clipped into the image).
        valid (np.ndarray): True while the step is inside the image and at or below top_row.
        exits_top (np.ndarray): Per ray, True if it leaves through the top_row edge rather than a side.
        reach_m (np.ndarray): Floor distance (m) from the ray's start point to each step.
        bearing_rad (np.ndarray): Direction from the camera to each step, positive = left.
        top_row (int): Highest image row the rays are allowed to reach.
        max_range_m (float): Floor distance that top_row corresponds to.
    """

    angles_rad: np.ndarray
    u: np.ndarray
    v: np.ndarray
    valid: np.ndarray
    exits_top: np.ndarray
    reach_m: np.ndarray
    bearing_rad: np.ndarray
    top_row: int
    max_range_m: float


@dataclass
class RayResult:
    """
    What each ray found on one frame.

    Attributes:
        reach_m (np.ndarray): Free floor distance (m) per ray.
        bearing_rad (np.ndarray): Direction (rad) of each ray's last free pixel, positive = left.
        is_open (np.ndarray): True if the ray made it all the way to top_row without hitting anything.
        end_u (np.ndarray): Column of each ray's last free pixel.
        end_v (np.ndarray): Row of each ray's last free pixel.
    """

    reach_m: np.ndarray
    bearing_rad: np.ndarray
    is_open: np.ndarray
    end_u: np.ndarray
    end_v: np.ndarray


# one ray per degree from -80 to +80 if the caller doesn't pick its own fan
DEFAULT_ANGLES_DEG = tuple(range(-80, 81))


def build_ray_table(intr: Intrinsics, pose: CameraPose, top_row: int, max_range_m: float,
                    angles_deg: Sequence[float] = DEFAULT_ANGLES_DEG) -> RayTable:
    """
    Precompute the fan of rays from the bottom-centre pixel. Only needs to run once.

    Args:
        intr (Intrinsics): Camera intrinsics.
        pose (CameraPose): Camera height and pitch.
        top_row (int): Rays stop when they go above this row.
        max_range_m (float): Floor distance at top_row, given to rays that get there unblocked.
        angles_deg (Sequence[float]): Image-space angles of the rays in degrees, 0 = straight up.

    Returns:
        RayTable: Pixel positions, floor distances and bearings for every step of every ray.
    """
    width, height = intr.width, intr.height
    angles_rad = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))

    # step one pixel at a time along each ray; width + height steps is more than any ray needs
    steps = np.arange(width + height, dtype=np.float64)
    u_raw = np.rint(intr.cx + steps[None, :] * np.sin(angles_rad)[:, None]).astype(np.int64)
    v_raw = np.rint((height - 1) - steps[None, :] * np.cos(angles_rad)[:, None]).astype(np.int64)

    inside = (u_raw >= 0) & (u_raw < width) & (v_raw >= 0) & (v_raw < height)
    valid = inside & (v_raw >= top_row)

    # trim the arrays to the longest ray (plus one invalid step so every ray has an end)
    n_steps = int(valid.sum(axis=1).max()) + 1
    u_raw, v_raw, valid = u_raw[:, :n_steps], v_raw[:, :n_steps], valid[:, :n_steps]

    # work out whether each ray leaves through the top (open road) or through a side
    first_invalid = np.argmax(~valid, axis=1)
    ray_ids = np.arange(len(angles_rad))
    exit_u = u_raw[ray_ids, first_invalid]
    exit_v = v_raw[ray_ids, first_invalid]
    exits_top = (exit_v < top_row) & (exit_u >= 0) & (exit_u < width)

    # clip so the lookups below never index outside the image
    u = np.clip(u_raw, 0, width - 1)
    v = np.clip(v_raw, 0, height - 1)

    # reach is measured on the floor from the ray's start point (the first step of the first ray,
    # which is the bottom-centre pixel); bearing is measured from the camera itself
    forward, left = pixel_to_floor(u, v, intr, pose)
    reach_m = np.where(valid, np.hypot(forward - forward[0, 0], left - left[0, 0]), 0.0)
    bearing_rad = np.where(valid, np.arctan2(left, forward), 0.0)

    return RayTable(angles_rad=angles_rad, u=u, v=v, valid=valid, exits_top=exits_top,
                    reach_m=reach_m, bearing_rad=bearing_rad, top_row=top_row,
                    max_range_m=max_range_m)


def cast_rays(table: RayTable, blocked: np.ndarray) -> RayResult:
    """
    Walk every ray until its first blocked pixel and report how far it got.

    Args:
        table (RayTable): Output of build_ray_table().
        blocked (np.ndarray): Boolean mask, True = the car's centre can't go here.

    Returns:
        RayResult: Reach, bearing and end pixel of every ray.
    """
    ray_ids = np.arange(table.u.shape[0])

    # a ray stops at the first step that is blocked or has left the usable part of the image
    stop = ~table.valid | blocked[table.v, table.u]
    first_stop = np.argmax(stop, axis=1)
    origin_blocked = first_stop == 0
    last_free = np.maximum(first_stop - 1, 0)

    # open = stopped only because it ran past top_row, not because it hit something
    is_open = ~table.valid[ray_ids, first_stop] & table.exits_top

    # open rays get the full range; the rest get the distance to their last free pixel
    reach_m = np.where(is_open, table.max_range_m, table.reach_m[ray_ids, last_free])
    # if the very first pixel is blocked there's no direction to report, so use 0
    bearing_rad = np.where(origin_blocked, 0.0, table.bearing_rad[ray_ids, last_free])

    return RayResult(reach_m=reach_m, bearing_rad=bearing_rad, is_open=is_open,
                     end_u=table.u[ray_ids, last_free], end_v=table.v[ray_ids, last_free])
