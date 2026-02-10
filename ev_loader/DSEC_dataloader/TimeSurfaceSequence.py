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
from ..representations.time_surface import ToTimesurface

class TimeSurfaceSequence(Sequence):
    def __init__(self, num_events: int = 10000, rep_subsample_factor: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.num_events = num_events  # Number of events per sample
        self.rep_subsample_factor = rep_subsample_factor
        self.events = self.event_slicers["left"].events
        self.to_timesurface = ToTimesurface(sensor_size=(self.width, self.height, 2), 
                                            surface_dimensions=None, 
                                            tau=5e3, 
                                            decay="exp"
                                            )
        
    def get_time_surface(self, x, y, p, t):
        # 1. Cast and Clip to prevent "double free or corruption"
        # We must ensure 0 <= x < width and 0 <= y < height
        int_x = np.clip(x.astype(np.int32), 0, self.width - 1)
        int_y = np.clip(y.astype(np.int32), 0, self.height - 1)
        int_p = np.clip(p.astype(np.int32), 0, 1) # Ensure polarity is 0 or 1

        surface_ref_indices = np.array([len(t) - 1]) if len(t) > 0 else np.array([0])  # Use the last event as reference for the surface
        events_dict = {
            'x': int_x,
            'y': int_y,
            'p': int_p,
            't': t
        }
        ts = self.to_timesurface(events_dict, surface_ref_indices) # (1, C, H, W)
        return torch.from_numpy(ts).float()
    
    def get_stacked_tsurface_representation(self, x, y, p, t):
        n_rep = x.shape[-1]//self.rep_subsample_factor

        sequence = []
        for seq_n in range(1, n_rep+1):
            seq_index = seq_n * self.rep_subsample_factor
            ts = self.get_time_surface(x[:seq_index], 
                                       y[:seq_index], 
                                       p[:seq_index], 
                                       t[:seq_index]
                                       )   # (1, C, H, W)
            sequence.append(ts)
        return torch.cat(sequence, dim=0)  # (seq_len, C, H, W)
    
    def __len__(self):
        return len(self.events['t'])//self.num_events  # Number of events in the sequence

    def __getitem__(self, index):
        id_start = index * self.num_events
        id_end = id_start + self.num_events

        if id_end > len(self.events['t']):
            id_end = len(self.events['t'])
            id_start = max(0, id_end - self.num_events)

        x = self.events['x'][id_start:id_end]
        y = self.events['y'][id_start:id_end]
        p = self.events['p'][id_start:id_end]
        t = self.events['t'][id_start:id_end]
        
        x_rect, y_rect = self.get_rectified_events(x, y)
        if self.rep_subsample_factor > 0:        
            rep = self.get_stacked_tsurface_representation(x=x_rect, y=y_rect, p=p, t=t)
        else:
            rep = self.get_time_surface(x=x_rect, y=y_rect, p=p, t=t) # (1, C, H, W)
            
        events = np.stack([x_rect, y_rect, p, t], axis=1)
        events_tensor = torch.from_numpy(events).float()

        data = Data(
            x=events_tensor,
            sequence_id=self.sequence_id, 
            event_representation=rep 
        )

        return data


if __name__ == '__main__':
    seq_abs_path = Path("/users/rpellerito/scratch/datasets/DSEC/test/zurich_city_14_c")
    dsec_seq = TimeSurfaceSequence(
        seq_path=seq_abs_path, 
        num_bins=2, 
        representation="stack", 
        num_events=50000, 
        mode='test',
        rep_subsample_factor=4999
        )

    # Use PyG DataLoader
    loader = DataLoader(dsec_seq, batch_size=1, shuffle=False)

    for batch in loader:
        # 'batch' is now a SINGLE Data object containing the concatenated data
        
        # This contains events from ALL 4 samples concatenated
        # Shape: [Total_N_Events_In_Batch, 4]
        all_events = batch.x 

        surface = batch["event_representation"]
        print(f"Time Surface Representation shape: {surface.shape}")  # (B, S, C, H, W)
        
        
        dense_x, mask = to_dense_batch(batch.x, batch.batch)
        # print(f"Dense x shape: {dense_x.shape}, Mask shape: {mask.shape}")

        dense_masked_x = dense_x * mask.unsqueeze(-1)
        # print(f"Dense Masked x shape: {dense_masked_x.shape}")

        # Check if the mask containts any zeros (it shouldn't in this case)
        if torch.any(mask == 0):
            print("Mask contains zeros!")
        else:
            pass