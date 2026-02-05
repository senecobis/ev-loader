"""
Modification of the official DSEC data loader https://github.com/uzh-rpg/DSEC/tree/main 
from Roberto Pellerito rpellerito@ifi.uzh.ch
"""
import torch
import numpy as np
from pathlib import Path
from .Sequence import Sequence
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from ..utils.VectorizedPatchfier import VectorizedPatchfier

class PatchedSequence(Sequence):
    def __init__(self, 
                 grid_size=(4, 4), 
                 events_per_patch=1024, 
                 **kwargs):
        super().__init__(**kwargs)
        self.grid_h, self.grid_w = grid_size
        self.k = events_per_patch
        self.H, self.W = kwargs.get('height', 256), kwargs.get('width', 256)
        
        # 1. Pre-sort all events into spatial buckets (The Indexing Phase)
        print("Indexing events into spatial buckets... this may take a moment.")
        raw_events = self.event_slicers["left"].events
        x, y = raw_events['x'], raw_events['y']
        
        # Calculate patch indices for every single event in the recording
        p_h, p_w = self.H // self.grid_h, self.W // self.grid_w
        px = (x // p_w).clip(0, self.grid_w - 1)
        py = (y // p_h).clip(0, self.grid_h - 1)
        patch_ids = (py * self.grid_w + px).astype(np.int32)

        # 2. Store indices for each patch separately
        self.patch_indices = []
        for i in range(self.grid_h * self.grid_w):
            # Find global indices of events belonging to this specific patch
            idxs = np.where(patch_ids == i)[0]
            self.patch_indices.append(idxs)
            
        # The length is determined by the "weakest" patch (the one with the fewest events)
        self.min_len = min([len(p) for p in self.patch_indices])
        print(f"Indexing complete. Shortest patch has {self.min_len} events.")

    def __len__(self):
        # How many chunks of 'k' events can we get from the shortest stream?
        return self.min_len // self.k

    def __getitem__(self, index):
        # Start and end relative to each patch's internal timeline
        start = index * self.k
        end = start + self.k
        
        all_patches = []
        raw_data = self.event_slicers["left"].events
        
        for i in range(self.grid_h * self.grid_w):
            # Get the global indices for THIS patch's current window
            global_idxs = self.patch_indices[i][start:end]
            
            # Pull the data
            px = raw_data['x'][global_idxs]
            py = raw_data['y'][global_idxs]
            pt = raw_data['t'][global_idxs]
            pp = raw_data['p'][global_idxs]
            
            # Localize coordinates
            off_x = (i % self.grid_w) * (self.W // self.grid_w)
            off_y = (i // self.grid_w) * (self.H // self.grid_h)
            
            patch_events = np.stack([px - off_x, py - off_y, pt, pp], axis=1)
            all_patches.append(patch_events)

        # Return shape: (Num_Patches, K, 4)
        # Every single patch is guaranteed to have exactly 'k' events.
        patched_tensor = torch.from_numpy(np.array(all_patches)).float()
        
        return Data(x=patched_tensor)