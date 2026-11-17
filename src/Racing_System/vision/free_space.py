"""
Floor segmentation for the vision follower.

Our track floor is light grey and the walls are black ducts, so we find free space by
colour: anything bright and grey is floor, everything else is treated as an obstacle.
The mask then gets cleaned up and every obstacle is grown sideways by half the car's
width, so a pixel left free means the centre of the car could actually drive there.

Masks in this module use True = not floor (or blocked) and False = floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FreeSpaceConfig:
    """
    Thresholds and kernel sizes used to build the floor mask.

    Attributes:
        v_floor_min (int): Minimum HSV value (brightness, 0-255) for a floor pixel.
        s_floor_max (int): Maximum HSV saturation (0-255) for a floor pixel.
        open_kernel (int): Size (px) of the square kernel that removes small specks.
        close_kernel (int): Size (px) of the square kernel that fills small holes in obstacles.
        min_vertical_px (int): Obstacles shorter than this many rows get removed (floor tape).
    """

    v_floor_min: int = 90
    s_floor_max: int = 40
    open_kernel: int = 3
    close_kernel: int = 7
    min_vertical_px: int = 6


def floor_threshold(rgb: np.ndarray, cfg: FreeSpaceConfig) -> np.ndarray:
    """
    Mark every pixel that doesn't look like floor.

    Args:
        rgb (np.ndarray): RGB image, shape (H, W, 3), uint8.
        cfg (FreeSpaceConfig): Thresholds to use.

    Returns:
        np.ndarray: Boolean mask, True = not floor.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # floor is bright (high V) and grey (low S); black ducts fail the V check
    is_floor = (hsv[..., 2] >= cfg.v_floor_min) & (hsv[..., 1] <= cfg.s_floor_max)
    return ~is_floor


def clean_mask(raw: np.ndarray, cfg: FreeSpaceConfig) -> np.ndarray:
    """
    Clean up the raw non-floor mask.

    Three passes: an open to get rid of small specks, a close to fill holes inside
    obstacles, then a tall thin open that removes anything only a few rows high
    (like tape lines on the floor).

    Args:
        raw (np.ndarray): Mask from floor_threshold(), True = not floor.
        cfg (FreeSpaceConfig): Kernel sizes to use.

    Returns:
        np.ndarray: Cleaned boolean mask, True = not floor.
    """
    speck_kernel = np.ones((cfg.open_kernel, cfg.open_kernel), np.uint8)
    hole_kernel = np.ones((cfg.close_kernel, cfg.close_kernel), np.uint8)
    # one pixel wide and min_vertical_px tall, so only short blobs get erased
    tape_kernel = np.ones((cfg.min_vertical_px, 1), np.uint8)

    mask = raw.astype(np.uint8)
    mask = _morph_floor_border(mask, cv2.MORPH_OPEN, speck_kernel)
    mask = _morph_floor_border(mask, cv2.MORPH_CLOSE, hole_kernel)
    mask = _morph_floor_border(mask, cv2.MORPH_OPEN, tape_kernel)
    return mask.astype(bool)


def _morph_floor_border(mask: np.ndarray, op: int, kernel: np.ndarray) -> np.ndarray:
    """
    Run a morphological open or close, pretending the image is surrounded by floor.

    OpenCV's default border handling can make obstacles appear or grow at the image
    edges, so we pad with floor (0) ourselves and crop it off afterwards.

    Args:
        mask (np.ndarray): uint8 mask, 1 = not floor.
        op (int): cv2.MORPH_OPEN or cv2.MORPH_CLOSE.
        kernel (np.ndarray): Structuring element.

    Returns:
        np.ndarray: uint8 mask, same shape as the input.

    Raises:
        ValueError: If op is anything other than open or close.
    """
    # open = erode then dilate, close = dilate then erode
    if op == cv2.MORPH_OPEN:
        first_step, second_step = cv2.erode, cv2.dilate
    elif op == cv2.MORPH_CLOSE:
        first_step, second_step = cv2.dilate, cv2.erode
    else:
        raise ValueError("only cv2.MORPH_OPEN and cv2.MORPH_CLOSE are supported")

    pad = max(kernel.shape)
    padded = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    # even-sized kernels have no true centre, so the second step uses the mirrored anchor
    # to land the result back where it started instead of drifting by a pixel
    kh, kw = kernel.shape
    anchor = (kw // 2, kh // 2)
    mirrored = (kw - 1 - kw // 2, kh - 1 - kh // 2)
    out = second_step(first_step(padded, kernel, anchor=anchor), kernel, anchor=mirrored)
    return out[pad:-pad, pad:-pad]


def nonfloor_mask(rgb: np.ndarray, cfg: FreeSpaceConfig) -> np.ndarray:
    """
    Threshold and clean in one call.

    Args:
        rgb (np.ndarray): RGB image, shape (H, W, 3), uint8.
        cfg (FreeSpaceConfig): Thresholds and kernel sizes.

    Returns:
        np.ndarray: Cleaned boolean mask, True = not floor.
    """
    return clean_mask(floor_threshold(rgb, cfg), cfg)


def push_obstacles(nonfloor: np.ndarray, push_px: np.ndarray, top_row: int) -> np.ndarray:
    """
    Grow every obstacle sideways by the car's half-width for that row.

    After this, a free pixel is somewhere the centre of the car could be without the
    sides hitting anything. Rows above top_row are never used, so they're left free.

    Args:
        nonfloor (np.ndarray): Cleaned mask, True = not floor.
        push_px (np.ndarray): Pixels to grow obstacles by, one value per row.
        top_row (int): First row to process.

    Returns:
        np.ndarray: Boolean mask, True = the car's centre can't go here.
    """
    height, width = nonfloor.shape
    blocked = np.zeros((height, width), dtype=bool)
    if top_row >= height:
        return blocked

    rows = nonfloor[top_row:]
    cols = np.arange(width)[None, :]
    # stand-in index for "no obstacle on this side", far enough away that it never counts
    no_obstacle = 10 * width

    # for every pixel, the column of the nearest obstacle to its left and to its right
    # (running max from the left, running min from the right)
    left_idx = np.maximum.accumulate(np.where(rows, cols, -no_obstacle), axis=1)
    right_idx = np.minimum.accumulate(np.where(rows, cols, no_obstacle)[:, ::-1], axis=1)[:, ::-1]
    distance = np.minimum(cols - left_idx, right_idx - cols)

    # blocked if the nearest obstacle in the row is within that row's push distance
    blocked[top_row:] = distance <= push_px[top_row:, None]
    return blocked
