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
    def __init__(self, num_events: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.num_events = num_events  # Number of events per sample
        self.max_n_objects = 20
        self.tracks_loader = TracksLoader(self.sequence_path, sync="back", timestamps_images=self.timestamps)
    
    def tracks2tensor(self, raw_tracks):
        # Convert tracks to a tensor format with fixed len
        tracks_tensor = torch.zeros((self.max_n_objects, 5), dtype=torch.float32) # 5 columns: [class_id, cx, cy, w, h]
        
        num_objects = len(raw_tracks)
        if num_objects > 0:
            for i, track in enumerate(raw_tracks[:self.max_n_objects]):
                class_id = track[5]
                x_tl = track[1]
                y_tl = track[2]
                w = track[4]
                h = track[3]
                tracks_tensor[i] = torch.tensor(
                    [class_id, x_tl, y_tl, w, h], 
                    dtype=torch.float32
                    )
        return tracks_tensor
    
    @staticmethod
    def top_left_to_center(tracks_tensor):
        # Convert from top-left to center format
        tracks_tensor[:, 1] += tracks_tensor[:, 3] / 2  # cx = x + w/2
        tracks_tensor[:, 2] += tracks_tensor[:, 4] / 2  # cy = y + h/2
        return tracks_tensor
    
    def __getitem__(self, index):
        ts_end = self.timestamps[index]
        ts_start = ts_end - self.delta_t_us
        x_rect, y_rect, p, t = self.get_rectified_events_start_end_time(ts_start, ts_end)
        if self.num_events > 0:
            #only load the most recent num_events events
            x_rect = x_rect[-self.num_events:]
            y_rect = y_rect[-self.num_events:]
            p = p[-self.num_events:]
            t = t[-self.num_events:]
        events = np.stack([x_rect, y_rect, p, t], axis=1)
        events_tensor = torch.from_numpy(events).float()
        
        # Dsec detections
        tracks = self.tracks_loader[index]
        tracks_tensor = self.tracks2tensor(tracks)
        tracks_tensor = self.top_left_to_center(tracks_tensor)

        data = Data(
            x=events_tensor,
            sequence_id=self.sequence_id, 
            tracks=tracks_tensor,
            num_actual_objects=len(tracks)
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