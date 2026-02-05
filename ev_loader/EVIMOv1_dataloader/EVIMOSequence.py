"""
Script to load EVIMO v1 dataset for training by Roberto Pellerito rpellerito@ifi.uzh.ch
"""

# TODO: add same agumentation of DSEC
# TODO: chnage to relative import ev_loader functions
# TODO: batch load events more efficently. Load all the events then collate them by slicing the events according to the maximum number of events in a batch

import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from ev_loader.DSEC_dataloader.Sequence import Sequence
from ev_loader.representations.representations import VoxelGrid
from ev_loader.utils.utils_events_split import split_events
from ..utils.utils_augmentations import EventListAugmentor


class EVIMOSequence(Dataset):
    def __init__(self, h5_path: str, window_ms=50.0, num_bins=2, sequence_len=1, batch_size=1, augment=[], augment_prob=[]):
        """
        PyTorch Dataset to load EVIMO1 .h5 files with structured events.
        
        Assumes HDF5 contains:
            - 'events': structured array with fields 'x', 'y', 't', 'p'
            - optionally: 'depth', 'classical', etc.
        """
        self.h5_path = str(h5_path)
        assert self.h5_path.endswith('.h5') or self.h5_path.endswith('.hdf5'), "Only .h5 files are supported."
        assert os.path.exists(self.h5_path), f"HDF5 file not found: {self.h5_path}"
        
        self.h5_file = h5py.File(self.h5_path, 'r')
        self.events = self.h5_file['events']
        self.index = self.h5_file['index']
        self.mask = self.h5_file['mask']
        self.timestamps = self.h5_file['ts']
        self.height = self.h5_file['height'][()]
        self.width = self.h5_file['width'][()]
        
        seq_name = self.h5_path.split('/')[-2]
        seq_segment = self.h5_path.split('/')[-1].split('.')[0]
        self.sequence_id = f"{seq_name}_{seq_segment}"

        events_t = self.h5_file['events_t']

        self.depth = self.h5_file['depth']
        self.meta = self.h5_file['meta']
        self.discretization = self.h5_file['discretization']
        self.fx = self.h5_file['fx']
        self.fy = self.h5_file['fy']
        self.cx = self.h5_file['cx']
        self.cy = self.h5_file['cy']

        # Simulation parameters
        self.window = window_ms / 1000.0  # ms → seconds
        self.num_bins = num_bins
        self.sequence_len = sequence_len
        self.max_num_grad_events = 50000
        self.max_num_detach_events = 50000
        self.batch_size = batch_size
        if self.batch_size == 1:
            self.max_num_grad_events = None
    
        start_timestamps = self.timestamps[()] - self.window
        start_timestamps = np.clip(start_timestamps, a_min=0, a_max=None)
        # self.start_timestamps = start_timestamps

        self.start_ev_ind = np.searchsorted(events_t, start_timestamps)
        self.end_ev_ind = np.searchsorted(events_t, self.timestamps)

        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=True)

        self.augment = augment
        self.augument_prob = augment_prob

        augmentations = {}
        for aug, aug_prob in zip(self.augment, self.augument_prob):
            if aug not in ["Horizontal", "Vertical", "Polarity"]:
                raise ValueError(f"Unknown augmentation mechanism: {aug}")
            if aug == "Horizontal":
                augmentations["hflip_prob"] = aug_prob
            elif aug == "Vertical":
                augmentations["vflip_prob"] = aug_prob
            elif aug == "Polarity":
                augmentations["polarity_flip_prob"] = aug_prob
        self.event_augment = EventListAugmentor(width=self.width, height=self.height, **augmentations)


    def search_closest_event_index(self, tstamp):
        """Return closest event index in the index array to a given timestamp"""
        closest_idx = np.argmin(np.abs(self.indices_tstamps - tstamp))
        return self.index[closest_idx]

    def __len__(self):
        return len(self.timestamps)
    
    def _events_to_voxel_grid(self, x, y, p, t):
        return Sequence.events_to_voxel_grid(self.voxel_grid, x, y, p, t)
    
    @staticmethod
    def convert_to_binary_mask(gt_mask):
        background_value = np.min(gt_mask)  # assuming background has the lowest value
        binary_mask = (gt_mask > background_value).astype(np.uint8)
        return binary_mask

    def get_single_item(self, idx):
        # Load by index
        with torch.no_grad():
            id0 = self.start_ev_ind[idx]
            id1 = self.end_ev_ind[idx]

            events = self.events[id0:id1]
            ev_x = events[:, 0]
            ev_y = events[:, 1]
            ev_t = events[:, 2]
            ev_p = events[:, 3]

            ev_x, ev_y, ev_p = self.event_augment(x=ev_x, y=ev_y, p=ev_p)
            
            dt = ev_t[-1] - ev_t[0]
            voxel = self._events_to_voxel_grid(x=ev_x, y=ev_y, p=ev_p, t=ev_t)

            events_, pol_mask_, d_events, d_pol_mask = split_events(
                x_rect=ev_x,
                y_rect=ev_y,
                p=ev_p,
                t=ev_t, 
                max_num_grad_events=self.max_num_grad_events, 
                max_num_detach_events=self.max_num_detach_events
                )
            
            binary_mask = self.convert_to_binary_mask(self.mask[idx])

            return {
                "representation": voxel,
                "dynamic_mask": torch.from_numpy(binary_mask).unsqueeze(0),
                "sampled_dt": torch.tensor([dt]),
                "event_list": events_,
                "polarity_mask":pol_mask_,
                "d_event_list": d_events,
                "d_polarity_mask": d_pol_mask,
            }
    
    def _get_sequence_indices(self, index):
        """
        Load the sequence whose start is at sequence_len steps index and
        end at index. If index is not the first index, in this case we load 
        the sequence starting at index and ending at index + sequence_len +1.
        """
        # <= to avoid loading index 0
        if index <= self.sequence_len +1:
            # if the first index is negative use the forward sequence
            indices = list(range(index, index + self.sequence_len + 1))
            return indices
        start_ind = index - self.sequence_len - 1
        indices = list(range(start_ind, index))
        return indices
    
    def __getitem__(self, idx):
        indices = self._get_sequence_indices(idx)
        sequence_ = []
        for index in indices:
            data = self.get_single_item(index)
            sequence_.append(data)
        return sequence_
    

if __name__ == "__main__":
    # Example usage
    # h5_path = "/home/rpg/Downloads/EVIMO1/train/box/seq_00.h5"
    h5_path = "/data/scratch/pellerito/datasets/EVIMO1/train/box/seq_00.h5"


    dataset = EVIMOSequence(h5_path, batch_size=2, window_ms=50, augment=["Horizontal", "Vertical"], augment_prob=[0.5, 0.5])
    print(f"Number of samples: {len(dataset)}")
    
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"Sample {i}: {sample}")