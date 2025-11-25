import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

def check_depth_event_density(depth_map, min_events):
    mask_start =  depth_map / 1000
    mask_start[mask_start > 0] = 1

    if np.nansum(mask_start) < min_events:
        return False
    return True

def full_mask_from_depth(depth_map_start, depth_map_end, num_bins):
    """
    Create a full mask from a depth map.
    
    Args:
        depth_map (np.ndarray): Depth map with shape [H, W, 3] where the third channel is depth.

    Returns:
        np.ndarray: Binary mask where depth > 0 is set to 1.
    """
    mask_start = depth_map_start / 1000
    mask_start[mask_start > 0] = 1
    mask_end = depth_map_end / 1000
    mask_end[mask_end > 0] = 1
    fullmask = np.stack([mask_start, mask_end], axis=0).max(axis=0)
    kernel = np.ones((5, 5), "uint8")
    fullmask = cv2.dilate(fullmask, kernel, iterations=1)
    full_mask_tensor = torch.from_numpy(
            np.tile(np.expand_dims(fullmask, axis=(0, 3)), (2, 1, 1, num_bins))
        )
    return full_mask_tensor

def get_masked_spike_tensor(spike_tensor, depth_map_start, depth_map_end, num_bins):
    """ Create a masked spike tensor based on depth maps.
    Args:
        spike_tensor (torch.Tensor): Tensor containing all spikes.
        depth_map_start (np.ndarray): Depth map at the start.
        depth_map_end (np.ndarray): Depth map at the end.
    Returns:
        torch.Tensor: Masked spike tensor where spikes are masked by depth maps.
    """
    full_mask_tensor = full_mask_from_depth(depth_map_start, depth_map_end, num_bins)
    masked_spike_tensor = ((spike_tensor + full_mask_tensor) > 1).float()
    return masked_spike_tensor

def background_ratio(spike_tensor, masked_spike_tensor):
    """
    Check if the ratio of background spikes to masked spikes exceeds a threshold.
    
    Args:
        spike_tensor (torch.Tensor): Tensor containing all spikes.
        masked_spike_tensor (torch.Tensor): Tensor containing masked spikes.
        max_background_ratio (float): Maximum allowed ratio of background spikes.

    Returns:
        bool: True if the ratio is within limits, False otherwise.
    """
    background_spikes = (spike_tensor + torch.logical_not(masked_spike_tensor).float()) > 1
    ratio = torch.sum(background_spikes) / torch.sum(masked_spike_tensor)
    return ratio


def piou_evimo(
    events, 
    pred_mask, 
    gt_mask, 
    depth_map_start, 
    depth_map_end, 
    num_bins: int = 100, 
    min_events: int = 30,
    max_background_ratio: float = 2.0):
    """
    Compute IoU between predicted and ground truth masks using event data.
    
    Args:
        events (torch.Tensor): [N, 4] tensor with [timestamp, x, y, polarity]
        pred_mask (np.ndarray or torch.Tensor): predicted binary mask (2D)
        gt_mask (np.ndarray or torch.Tensor): groundtruth binary mask (2D)
        depth_map_start (np.ndarray, optional): 
        depth_map_end (np.ndarray, optional):
        num_bins (int): temporal bins used to discretize event timestamps

    Returns:
        float: IoU value between predicted and ground-truth masks
    """
    # Validate input shapes
    assert pred_mask.shape == gt_mask.shape, "Mask dimensions must match"
    
    # Check if enough events are associated with depth maps
    enough_density_start = check_depth_event_density(depth_map_start, min_events) 
    enough_density_end = check_depth_event_density(depth_map_end, min_events)
    if not (enough_density_start or enough_density_end):
        return np.nan

    height, width = gt_mask.shape

    # Normalize timestamps into bins
    xs = events[:, 0].numpy().astype(np.int64)
    ys = events[:, 1].numpy().astype(np.int64)
    ts = events[:, 2].numpy()
    ts = ((num_bins - 1) * (ts - ts[0]) / (ts[-1] - ts[0])).astype(np.int64)
    ps = events[:, 3].numpy().clip(0, 1).astype(np.int64)

    spike_tensor = torch.zeros((2, height, width, num_bins))
    spike_tensor[ps, ys, xs, ts] = 1
    
    masked_spike_tensor = get_masked_spike_tensor(
                                                spike_tensor, 
                                                depth_map_start, 
                                                depth_map_end, 
                                                num_bins=num_bins
                                                )
    ratio = background_ratio(spike_tensor, masked_spike_tensor)
    if ratio > max_background_ratio:
        return np.nan

    spike_mask_2D = torch.sum(masked_spike_tensor, dim=(0, 3))

    spike_pred_2D = pred_mask.unsqueeze(0).unsqueeze(-1).repeat(2, 1, 1, num_bins)
    spike_pred_2D[spike_pred_2D >= 0.5] = 1.0
    spike_pred_2D[spike_pred_2D < 0.5] = 0.0
    spike_pred_2D = ((spike_tensor + spike_pred_2D) > 1).float()
    spike_pred_2D = torch.sum(spike_pred_2D, dim=(0, 3))

    spike_pred = spike_pred_2D.numpy()
    spike_gt = spike_mask_2D.numpy()

    intersection = np.sum(np.logical_and(spike_pred, spike_gt))
    union = np.sum(np.logical_or(spike_pred, spike_gt))
    point_iou = intersection / union
    return point_iou


if __name__ == "__main__":
    num_bins = 100
    height = 260
    width = 346
    min_events = 30
    max_background_ratio = 2
    
    events_x = np.random.randint(0, width, 1000)
    events_y = np.random.randint(0, height, 1000)
    events_ts = np.sort(np.random.rand(1000) * 1000)  # Random timestamps
    events_p = np.random.randint(0, 2, 1000)  # Random polarities (0 or 1)
    events = np.stack((events_ts, events_x, events_y, events_p), axis=-1)
    events = torch.tensor(events, dtype=torch.float32)
    pred_mask = torch.rand(height, width) > 0.5  # Random predicted mask
    gt_mask = torch.rand(height, width) > 0.5  # Random ground truth mask
    depth_map_start = np.random.rand(height, width)
    depth_map_end = np.random.rand(height, width)
    print("Events shape:", events.shape)
    print("Pred mask shape:", pred_mask.shape)
    print("GT mask shape:", gt_mask.shape)
    print("Depth map start shape:", depth_map_start.shape)
    print("Depth map end shape:", depth_map_end.shape)
    piou = piou_evimo(
        events=events,
        pred_mask=pred_mask,
        gt_mask=gt_mask,
        depth_map_start=depth_map_start,
        depth_map_end=depth_map_end,
        num_bins=num_bins,
        min_events=min_events,
        max_background_ratio=max_background_ratio
    )
    print("Point IoU:", piou)