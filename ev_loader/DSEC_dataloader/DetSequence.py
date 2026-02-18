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
from .detection.directory import TracksLoader

class DetSequence(Sequence):
    def __init__(self, num_events: int = <, **kwargs):
        super().__init__(**kwargs)
        self.num_events = num_events  # Number of events per sample
        self.tracks_loader = TracksLoader(self.sequence_path, sync="back", timestamps_images=self.timestamps)
    
    def __getitem__(self, index):
        ts_end = self.timestamps[index]
        ts_start = ts_end - self.delta_t_us
        x_rect, y_rect, p, t = self.get_rectified_events_start_end_time(ts_start, ts_end)
        events = np.stack([x_rect, y_rect, p, t], axis=1)
        events_tensor = torch.from_numpy(events).float()
        
        # Dsec detections
        tracks = self.tracks_loader[index]

        data = Data(
            x=events_tensor,
            sequence_id=self.sequence_id, 
            tracks=tracks
        )
        return data


if __name__ == '__main__':
    seq_abs_path = Path("/users/rpellerito/scratch/datasets/DSEC/test/interlaken_00_a")
    dsec_seq = DetSequence(
        seq_path=seq_abs_path, 
        num_bins=2, 
        representation="stack", 
        num_events=50000, 
        mode='test',
        )

    # Use PyG DataLoader
    loader = DataLoader(dsec_seq, batch_size=10, shuffle=False)

    for batch in loader:
        # 'batch' is now a SINGLE Data object containing the concatenated data
        
        # This contains events from ALL 4 samples concatenated
        # Shape: [Total_N_Events_In_Batch, 4]
        all_events = batch.x 

        tracks = batch["tracks"]
        print(f"Tracks shape: {len(tracks)}")
        
        
        dense_x, mask = to_dense_batch(batch.x, batch.batch)
        # print(f"Dense x shape: {dense_x.shape}, Mask shape: {mask.shape}")

        dense_masked_x = dense_x * mask.unsqueeze(-1)
        # print(f"Dense Masked x shape: {dense_masked_x.shape}")

        # Check if the mask containts any zeros (it shouldn't in this case)
        if torch.any(mask == 0):
            print("Mask contains zeros!")
        else:
            pass