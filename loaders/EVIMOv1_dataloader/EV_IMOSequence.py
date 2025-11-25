"""
Script to load EVIMO v1 dataset for training by Roberto Pellerito rpellerito@ifi.uzh.ch
"""

# TODO: add same agumentation of DSEC
# TODO: chnage to relative import suppressor functions
# TODO: batch load events more efficently. Load all the events then collate them by slicing the events according to the maximum number of events in a batch

import torch
import numpy as np
from tqdm import tqdm
from .EVIMOSequence import EVIMOSequence
from ..utils.representations import events_to_channels, MeanTimestamps


class EV_IMOSequence(EVIMOSequence):
    def __init__(self, h5_path: str, window_ms, num_bins, sequence_len, dt=0, mask_future_events=False):
        super().__init__(h5_path, window_ms, num_bins, sequence_len)
        self.mean_tstamp = MeanTimestamps(self.height, self.width, smooth=False)
        self.dt = dt
        self.mask_future_events = mask_future_events

        self.depth = self.h5_file['depth']
        self.fx = self.h5_file['fx'][()]
        self.fy = self.h5_file['fy'][()]
        self.cx = self.h5_file['cx'][()]
        self.cy = self.h5_file['cy'][()]

    def events_to_mean_tstamp(self, x, y, t):
        if not isinstance(t, torch.Tensor):
            x = torch.from_numpy(x.astype(np.int64))
            y = torch.from_numpy(y.astype(np.int64))
            t = torch.from_numpy(t.astype(np.float32))
        return self.mean_tstamp.convert(x=x, y=y, t=t)
    
    def events_cnt_tstamp_representation(self, x, y, p, t):
        """transform events to stack of count image for positive and negative events
        and mean timestamp image.
        Args:
            x (_type_): _description_
            y (_type_): _description_
            p (_type_): _description_
            t (_type_): _description_
        """
        sensor_size = (self.height, self.width)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        p = torch.from_numpy(p)
        t = torch.from_numpy(t)
        p[p==0] = -1
        cnt_pos_neg = events_to_channels(xs=x, ys=y, ps=p, sensor_size=sensor_size)
        tstamp_image = self.events_to_mean_tstamp(x=x, y=y, t=t)
        representation_ = torch.cat((cnt_pos_neg, tstamp_image), dim=0)
        return representation_
    

    def representation_from_indices(self, id0, id1):
        events = self.events[id0:id1]
        ev_x = events[:, 0]
        ev_y = events[:, 1]
        ev_t = events[:, 2]
        ev_p = events[:, 3]

        # ev_x, ev_y, ev_p = self.event_augment(x=ev_x, y=ev_y, p=ev_p)
        cnt_tstamp = self.events_cnt_tstamp_representation(
            x=ev_x, y=ev_y, p=ev_p, t=ev_t
        )
        return cnt_tstamp
    
    @property
    def instrinsics(self):
        K = np.array([
            [self.fx, 0,  self.cx],
            [0,  self.fy, self.cy],
            [0,  0,  1]
        ])
        return K.astype(np.float32)

    @property
    def inv_intrinsics(self):
        """Invert the intrinsics matrix."""
        K_inv = np.linalg.inv(self.instrinsics)
        return K_inv
    
    def depth_normalization(self, depth):
        depth[np.isnan(depth)] = 10000
        depth[depth<100]=10000
        depth = depth.astype(np.float32) / 6000*255
        return depth

    def mask_normalization(self, obj_mask):
        obj_mask = obj_mask.astype(np.float32)/1000*255
        return obj_mask
    
    def get_single_item(self, idx):
        # Load by index
        with torch.no_grad():
            id0 = self.start_ev_ind[idx]
            id1 = self.end_ev_ind[idx]

            original_mask = self.mask[idx]
            binary_mask = self.convert_to_binary_mask(original_mask).astype(np.float32)
            depth = self.depth[idx]

            cnt_tstamp = self.representation_from_indices(id0, id1)

            reference_ev_frames = []
            ref_indices = self._get_ref_indices(idx)
            for ref_idx in ref_indices:
                id0_ref = self.start_ev_ind[ref_idx]
                id1_ref = self.end_ev_ind[ref_idx]
                cnt_tstamp_ = self.representation_from_indices(id0_ref, id1_ref)
                reference_ev_frames.append(cnt_tstamp_)

            mask_norm = self.mask_normalization(original_mask)
            depth_norm = self.depth_normalization(depth)

            if self.dt == 0:
                mask_future = original_mask
                mask_norm_future = mask_norm
                depth_norm_future = depth_norm
            else:
                mask_future = self.mask[idx+self.dt]
                mask_norm_future = self.mask_normalization(mask_future)
                depth_norm_future = self.depth_normalization(self.depth[idx+self.dt])

            return {
                "sequence_id": self.sequence_id,
                "target_event_representation": cnt_tstamp,
                "reference_event_representation": reference_ev_frames,
                "mask": original_mask,
                "dynamic_mask": torch.from_numpy(binary_mask).unsqueeze(0),
                "intrinsics": torch.from_numpy(self.instrinsics),
                "inverse_intrinsics": torch.from_numpy(self.inv_intrinsics),
                "normalized_mask": torch.from_numpy(mask_norm).unsqueeze(0),
                "normalized_depth": torch.from_numpy(depth_norm).unsqueeze(0),
                "normalized_mask_future": torch.from_numpy(mask_norm_future).unsqueeze(0),
                "normalized_depth_future": torch.from_numpy(depth_norm_future).unsqueeze(0),
                "mask_future": torch.from_numpy(mask_future).unsqueeze(0),
                "events":  self.events[id0:id1]
            }
            
    def _get_ref_indices(self, index):
        """
        Load the sequence whose start is at index-sequence_len and
        end at index.
        """
        if self.mask_future_events:
            return self._get_preceding_reference_indices(index)
        else:
            return self._get_centered_reference_indices(index)
    
    def _get_centered_reference_indices(self, index):
        """
        Load the sequence whose start is at sequence_len steps index and
        end at index. If index is not the first index, in this case we load 
        the sequence starting at index and ending at index + sequence_len +1.
        """
        # NOTE> this assumes index > sequence_len

        demi_length = (self.sequence_len-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        indices = [index + shift for shift in shifts]
        return indices
    
    def _get_preceding_reference_indices(self, index):
        """
        Load the sequence whose start is at index-sequence_len and
        end at index.
        """
        # NOTE> this assumes index > sequence_len
        # Last index (shift=0) is the target index
        indices = list(range(index-self.sequence_len, index-1))
        return indices
        
    def __len__(self):
        """Len of the dataset, it is subjected to change depending on 
           the sequence length, the dt but also if we load centered reference frames
           or preceding reference frames.

        Returns:
            int: len of the dataset
        """
        if self.mask_future_events:
            valid_start = self.sequence_len
            valid_end = super().__len__() - self.dt
            return valid_end - valid_start
        else:
            valid_start = (self.sequence_len - 1) // 2
            valid_end = super().__len__() - valid_start
            return valid_end - valid_start - self.dt
    
    def __getitem__(self, idx):
        if self.mask_future_events:
            real_idx = idx + self.sequence_len
        else:
            real_idx = idx + (self.sequence_len - 1) // 2
        return self.get_single_item(real_idx)

    

if __name__ == "__main__":
    # Example usage
    h5_path = "/data/scratch/pellerito/datasets/EVIMO1/train/box/seq_00.h5"

    dataset = EV_IMOSequence(h5_path, window_ms=50, num_bins=2, sequence_len=5, dt=1, mask_future_events=False)
    print(f"Number of samples: {len(dataset)}")
    
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        # print(f"Sample {i}: {sample}")