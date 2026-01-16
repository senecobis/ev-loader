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

class ByEvIdxSequence(Sequence):
    def __init__(self, num_events: int = 10000, **kwargs):
        super().__init__(**kwargs)
        self.num_events = num_events  # Number of events per sample
        self.events = self.event_slicers["left"].events
    
    def __len__(self):
        return len(self.events['t'])//self.num_events  # Number of events in the sequence

    def __getitem__(self, index):
        id_start = index * self.num_events
        id_end = id_start + self.num_events

        x = self.events['x'][id_start:id_end]
        y = self.events['y'][id_start:id_end]
        p = self.events['p'][id_start:id_end]
        t = self.events['t'][id_start:id_end]

        x_rect, y_rect = self.get_rectified_events(x, y)        
        event_representation = self.get_event_representation(x_rect, y_rect, p, t)

        down_events = np.stack([x_rect, y_rect, p, t], axis=1)
        events_tensor = torch.from_numpy(down_events).float()

        data = Data(
            x=events_tensor,
            sequence_id=self.sequence_id, 
            event_representation=event_representation.unsqueeze(0) 
        )

        return data


if __name__ == '__main__':

    seq_abs_path = Path("/data/scratch/pellerito/datasets/DSEC/test/zurich_city_14_c")
    dsec_seq = ByEvIdxSequence(seq_path=seq_abs_path, num_bins=2, representation="stack", num_events=50000, mode='test', delta_t_ms=50)

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
        
        print(f"\nTotal events in batch: {all_events.shape[0]}")

        dense_x, mask = to_dense_batch(batch.x, batch.batch)
        print(f"Dense x shape: {dense_x.shape}, Mask shape: {mask.shape}")

        dense_masked_x = dense_x * mask.unsqueeze(-1)
        print(f"Dense Masked x shape: {dense_masked_x.shape}")

        # Check if the mask containts any zeros (it shouldn't in this case)
        if torch.any(mask == 0):
            print("Mask contains zeros!")
        else:
            pass
