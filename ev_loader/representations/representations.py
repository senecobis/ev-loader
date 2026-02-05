import cv2
import torch
import numpy as np
from ev_loader.data import Events
from torchvision.transforms import GaussianBlur


class EventRepresentation:
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        raise NotImplementedError


class VoxelGrid(EventRepresentation):
    def __init__(self, channels: int, height: int, width: int, normalize: bool):
        self.voxel_grid = torch.zeros((channels, height, width), dtype=torch.float, requires_grad=False)
        self.nb_channels = channels
        self.normalize = normalize

    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(pol.device)
            voxel_grid = self.voxel_grid.clone()

            t_norm = time
            t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])

            x0 = x.int()
            y0 = y.int()
            t0 = t_norm.int()

            value = 2*pol-1

            for xlim in [x0,x0+1]:
                for ylim in [y0,y0+1]:
                    for tlim in [t0,t0+1]:

                        mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < self.nb_channels)
                        interp_weights = value * (1 - (xlim-x).abs()) * (1 - (ylim-y).abs()) * (1 - (tlim - t_norm).abs())

                        index = H * W * tlim.long() + \
                                W * ylim.long() + \
                                xlim.long()

                        voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

            if self.normalize:
                mask = torch.nonzero(voxel_grid, as_tuple=True)
                if mask[0].size()[0] > 0:
                    mean = voxel_grid[mask].mean()
                    std = voxel_grid[mask].std()
                    if std > 0:
                        voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                    else:
                        voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid


class EventToStack_Numpy(object):
    def __init__(self, num_bins, height, width):
        self.num_bins = num_bins
        self.height = height
        self.width = width

    def _draw_xy_to_voxel_grid(self, voxel_grid, x, y, b, value):
        if x.dtype == np.uint16:
            self._draw_xy_to_voxel_grid_int(voxel_grid, x, y, b, value)
            return

        x_int = x.astype("int32")
        y_int = y.astype("int32")
        for xlim in [x_int, x_int + 1]:
            for ylim in [y_int, y_int + 1]:
                weight = _bil_w(x, xlim) * _bil_w(y, ylim)
                self._draw_xy_to_voxel_grid_int(voxel_grid, xlim, ylim, b, weight * value)

    def _draw_xy_to_voxel_grid_int(self, voxel_grid, x, y, b, value):
        B, H, W = voxel_grid.shape
        mask = (x >= 0) & (y >= 0) & (x < W) & (y < H)
        np.add.at(voxel_grid, (b[mask], y[mask], x[mask]), value[mask])

    def __call__(self, x, y, p) -> np.array:
        x = x.astype(np.uint16)
        y = y.astype(np.uint16)
        voxel_grid = np.zeros((self.num_bins, self.height, self.width), np.float32)

        events_num = len(x)
        if events_num < 2:
            return voxel_grid

        # normalize the event timestamps so that they lie between 0 and num_bins
        t_norm = (self.num_bins * np.arange(events_num, dtype="float32") / events_num).astype("int32")
        self._draw_xy_to_voxel_grid(voxel_grid, x, y, t_norm, p)

        return voxel_grid.astype("int8")


class MeanTimestamps(object):
    def __init__(self, height: int, width: int, smooth: bool):
        self.H = height
        self.W = width
        self.smooth = smooth
            
    def convert(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        """
        Computes the normalized mean timestamp image using PyTorch.
        
        Parameters:
        x: Tensor of shape (N,) with x-coordinates of events.
        y: Tensor of shape (N,) with y-coordinates of events.
        t: Tensor of shape (N,) with timestamps of events.
        
        Returns:
        norm_mean_ts: Tensor of shape (H, W) with values normalized between 0 and 1.
        """
        
        device = t.device  # ensure everything is on the same device
        N = len(t)
        
        # normalise the timestamps to [0, 1]
        dt = (t[-1] - t[0]).item()
        if dt == 0:
            t = torch.zeros_like(t)
        else:
            t = (t - t[0]) / dt
        
        # mask the events outside the image plane
        mask_ = (x >= 0) & (x < self.W) & (y >= 0) & (y < self.H)
        x = x[mask_].long()
        y = y[mask_].long()
        t = t[mask_]
        
        # Flattened index for each event in a (H x W) image.
        indices = y * self.W + x  # shape: (N,)
        
        # Create tensors to accumulate the timestamp sums and counts.
        ts_sum_flat = torch.zeros(self.H * self.W, dtype=t.dtype, device=device)
        count_flat = torch.zeros(self.H * self.W, dtype=torch.int32, device=device)
        
        # Use scatter_add to accumulate values.
        ts_sum_flat = ts_sum_flat.scatter_add(0, indices, t)
        ones = torch.ones_like(t, dtype=torch.int32)
        count_flat = count_flat.scatter_add(0, indices, ones)
        
        # Reshape to (H, W)
        ts_sum = ts_sum_flat.view(self.H, self.W)
        count = count_flat.view(self.H, self.W)
        
        # Compute mean timestamps; avoid division by zero.
        mean_ts = torch.zeros((self.H, self.W), dtype=t.dtype, device=device)
        mask = count > 0
        mean_ts[mask] = ts_sum[mask] / count[mask].float()
        
        if self.smooth:
            # Smooth to avoid the checkerboard pattern from event rectification
            mean_ts = mean_ts.unsqueeze(0).unsqueeze(0)
            mean_ts = GaussianBlur(kernel_size=3, sigma=1.0)(mean_ts)
            mean_ts = mean_ts.squeeze(0).squeeze(0)
            
        return mean_ts.unsqueeze(0)

"""
Adapted from Monash University https://github.com/TimoStoff/events_contrast_maximization
"""

def events_to_channels(xs, ys, ps, sensor_size=(180, 240)):
    """
    Generate a two-channel event image containing per-pixel event counters.
    :param xs: event x coordinates
    :param ys: event y coordinates
    :param ps: event polarity
    :param sensor_size: sensor size
    :return: event image containing per-pixel and per-polarity event counts
    """

    assert len(xs) == len(ys) and len(ys) == len(ps)

    mask_pos = ps.clone()
    mask_neg = ps.clone()
    mask_pos[ps < 0] = 0
    mask_neg[ps > 0] = 0
    mask_pos[ps > 0] = 1
    mask_neg[ps < 0] = -1

    pos_cnt = events_to_image(xs, ys, ps * mask_pos, sensor_size=sensor_size)
    neg_cnt = events_to_image(xs, ys, ps * mask_neg, sensor_size=sensor_size)

    return torch.stack([pos_cnt, neg_cnt])


def create_cnt_encoding(xs, ys, ps, rect_mapping, sensor_size, device):
    """
    Creates a per-pixel and per-polarity event count representation.
    :param xs: [N] tensor with event x location
    :param ys: [N] tensor with event y location
    :param ps: [N] tensor with event polarity ([-1, 1])
    :param rect_mapping: map used to rectify events
    :return [2 x H x W] rectified event count representation
    """

    # create event count representation and rectify it using backward mapping
    event_cnt = events_to_channels(xs, ys, ps, sensor_size=sensor_size)
    if rect_mapping is not None:
        event_cnt = event_cnt.permute(1, 2, 0)
        event_cnt = cv2.remap(event_cnt.cpu().numpy(), rect_mapping, None, cv2.INTER_NEAREST)
        event_cnt = torch.from_numpy(event_cnt.astype(np.float32)).to(device)
        event_cnt = event_cnt.permute(2, 0, 1)

    return event_cnt

def events_to_image(xs, ys, ps, sensor_size=(180, 240), accumulate=True):
    """
    Accumulate events into an image.
    :param xs: event x coordinates
    :param ys: event y coordinates
    :param ps: event polarity
    :param sensor_size: sensor size
    :param accumulate: flag indicating whether to accumulate events into the image
    :return img: image containing per-pixel event counts
    """

    device = xs.device
    img_size = list(sensor_size)
    img = torch.zeros(img_size, device=device)

    if xs.dtype is not torch.long:
        xs = xs.long().to(device)
    if ys.dtype is not torch.long:
        ys = ys.long().to(device)
    img.index_put_((ys, xs), ps, accumulate=accumulate)

    return img
