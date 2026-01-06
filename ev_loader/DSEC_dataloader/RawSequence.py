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

class RawSequence(Sequence):
    def __init__(self, num_events: int = 10000, **kwargs):
        super().__init__(**kwargs)
        self.num_events = num_events  # Number of events per sample

    def __getitem__(self, index):
        ts_end = self.timestamps[index]
        ts_start = ts_end - self.delta_t_us

        file_index = self.file_index(index)
        frame = self.frame_gt(index)
        
        x_rect, y_rect, p, t = self.get_rectified_events_start_end_time(ts_start, ts_end)
        self.x_rect = x_rect
        self.y_rect = y_rect
        self.p = p
        self.t = t

        event_representation = self.get_event_representation(x_rect, y_rect, p, t)

        # Downsample events
        N = len(x_rect)
        if N > self.num_events:
            indices = np.random.choice(N, self.num_events, replace=False)
            indices.sort()
        else:
            indices = np.arange(N)

        down_events = np.stack([
            x_rect[indices], 
            y_rect[indices], 
            p[indices], 
            t[indices]
            ], axis=1)
        events_tensor = torch.from_numpy(down_events).float()
        

        data = Data(
            x=events_tensor,
            file_index=torch.tensor(file_index), 
            sequence_id=self.sequence_id, 
            frame=frame.unsqueeze(0),
            event_representation=event_representation.unsqueeze(0) 
        )

        return data


if __name__ == '__main__':

    seq_abs_path = Path("/data/scratch/pellerito/datasets/DSEC/test/zurich_city_14_c")
    dsec_seq = RawSequence(seq_path=seq_abs_path, num_bins=2, representation="stack")

    # Use PyG DataLoader
    # batch_size=4 means it will grab 4 samples and merge their events
    loader = DataLoader(dsec_seq, batch_size=4, shuffle=True)

    for batch in loader:
        # 'batch' is now a SINGLE Data object containing the concatenated data
        
        # This contains events from ALL 4 samples concatenated
        # Shape: [Total_N_Events_In_Batch, 4]
        all_events = batch.x 
        
        # This is the "Trick": A vector of shape [Total_N_Events_In_Batch]
        # It tells you which sample in the batch (0, 1, 2, or 3) the event belongs to.
        batch_indices = batch.batch 
        
        print(f"Total events in batch: {all_events.shape[0]}")
        print(f"Batch Vector: {batch_indices}")
        
        # Accessing the images (PyG stacks fixed size attributes normally)
        # Shape: [4, C, H, W]
        frames = batch.frame
        print(f"Frames shape: {frames.shape}")

        dense_x, mask = to_dense_batch(batch.x, batch.batch)
        print(f"Dense x shape: {dense_x.shape}, Mask shape: {mask.shape}")

        dense_masked_x = dense_x * mask.unsqueeze(-1)
        print(f"Dense Masked x shape: {dense_masked_x.shape}")
