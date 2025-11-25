"""
Adapted script from DSEC/scripts/check_optical_flow_submission.py

"""

import sys
version = sys.version_info
assert version[0] >= 3, 'Python 2 is not supported'
assert version[1] >= 6, 'Requires Python 3.6 or higher'

import os
os.environ['IMAGEIO_USERDIR'] = '/var/tmp'
from pathlib import Path
from enum import Enum, auto

import imageio
imageio.plugins.freeimage.download()
import numpy as np

import torch
import torch.nn.functional as F

def warp_flow(flow, flow_ref):
    """
    Warps `flow` using `flow_ref`. Supports batched input.

    Args:
        flow: Tensor of shape (B, 2, H, W) – the flow to warp (e.g., flow_1to2)
        flow_ref: Tensor of shape (B, 2, H, W) – the reference flow (e.g., flow_0to1)
    
    Returns:
        Tensor of shape (B, 2, H, W) – the warped flow
    """
    B, _, H, W = flow.shape

    # Create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=flow.device),
        torch.arange(W, device=flow.device),
        indexing='ij'
    )
    grid = torch.stack((grid_x, grid_y), dim=0).float()  # (2, H, W)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, H, W)

    # Compute sampling locations in t1
    coords = grid + flow_ref  # (B, 2, H, W)

    # Normalize to [-1, 1] for grid_sample
    coords_norm = torch.stack([
        2.0 * coords[:, 0] / (W - 1) - 1.0,
        2.0 * coords[:, 1] / (H - 1) - 1.0
    ], dim=-1)  # (B, H, W, 2)

    # Sample flow at warped coordinates
    warped_flow = F.grid_sample(flow, coords_norm, mode='bilinear', padding_mode='border', align_corners=True)

    return warped_flow

def sum_flows(flow_0to1, flow_1to2):
    """
    Composes flow_0to1 and flow_1to2 to get flow_0to2. Supports batched input.

    Args:
        flow_0to1: Tensor of shape (B, 2, H, W)
        flow_1to2: Tensor of shape (B, 2, H, W)

    Returns:
        Tensor of shape (B, 2, H, W) representing flow_0to2
    """
    flow_1to2_warped = warp_flow(flow_1to2, flow_0to1)
    flow_0to2 = flow_0to1 + flow_1to2_warped
    return flow_0to2

class WriteFormat(Enum):
    OPENCV = auto()
    IMAGEIO = auto()

def flow_16bit_to_float(flow_16bit: np.ndarray, valid_in_3rd_channel: bool):
    assert flow_16bit.dtype == np.uint16
    assert flow_16bit.ndim == 3
    h, w, c = flow_16bit.shape
    assert c == 3

    if valid_in_3rd_channel:
        valid2D = flow_16bit[..., 2] == 1
        assert valid2D.shape == (h, w)
        assert np.all(flow_16bit[~valid2D, -1] == 0)
    else:
        valid2D = np.ones_like(flow_16bit[..., 2], dtype=bool)
    valid_map = np.where(valid2D)

    # to actually compute something useful:
    flow_16bit = flow_16bit.astype('float')

    flow_map = np.zeros((h, w, 2))
    flow_map[valid_map[0], valid_map[1], 0] = (flow_16bit[valid_map[0], valid_map[1], 0] - 2**15) / 128
    flow_map[valid_map[0], valid_map[1], 1] = (flow_16bit[valid_map[0], valid_map[1], 1] - 2**15) / 128
    return flow_map, valid2D


def load_flow(flowfile: Path, valid_in_3rd_channel: bool = False, write_format=WriteFormat.OPENCV):
    # imageio reading assumes write format was rgb
    flow_16bit = imageio.imread(str(flowfile), format='PNG-FI')
    if write_format == WriteFormat.OPENCV:
        # opencv writes as bgr -> flip last axis to get rgb
        flow_16bit = np.flip(flow_16bit, axis=-1)
    else:
        assert write_format == WriteFormat.IMAGEIO

    channel3 = flow_16bit[..., -1]
    assert channel3.max() <= 1, f'Maximum value in last channel should be 1: {flowfile}'
    flow, _ = flow_16bit_to_float(flow_16bit, valid_in_3rd_channel)
    return flow