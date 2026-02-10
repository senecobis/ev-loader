import h5py
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from .Sequence import Sequence
from ..representations.time_surface import ToTimesurface


class IndexedPatchSequence(Sequence):
    def __init__(self, 
                 n_patches_h=4, 
                 n_patches_w=4, 
                 num_events=10000, 
                 rep_subsample_factor: int = 100, 
                 **kwargs):
        super().__init__(**kwargs)
        self.S = num_events
        self.n_h, self.n_w = n_patches_h, n_patches_w
        self.num_patches = n_patches_h * n_patches_w
        self.patch_h = self.height // n_patches_h
        self.patch_w = self.width // n_patches_w
        self.rep_subsample_factor = rep_subsample_factor
        self.patch_h5_path = self.sequence_path / "events_patchified.h5"

        self.events = self.event_slicers["left"].events
        
        if not self.patch_h5_path.exists():
            self.generate_patchified_h5()
        else:
            self.patch_ev = h5py.File(str(self.patch_h5_path), 'r')
            n_h = self.patch_ev.attrs['n_patches_h']
            n_w = self.patch_ev.attrs['n_patches_w']
            self.patch_ev.close()
            if n_h != self.n_h or n_w != self.n_w:
                print(f"Warning: Existing patchified H5 has different patch grid ({n_h}x{n_w}) than requested ({self.n_h}x{self.n_w}). Regenerating...")
                self.generate_patchified_h5()
        
        self.patch_h5 = h5py.File(str(self.patch_h5_path), 'r')

        self.patch_offsets = [] 
        for p_id in range(self.num_patches):
            n_ev = self.patch_h5[f"patch_{p_id}/t"].shape[0]
            
            num_slices_in_patch = n_ev // self.S
            
            for i in range(num_slices_in_patch):
                self.patch_offsets.append((p_id, i * self.S))
        
        self.total_len = len(self.patch_offsets)

        self.to_timesurface = ToTimesurface(sensor_size=(self.patch_w, self.patch_h, 2), 
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
        return self.total_len

    def generate_patchified_h5(self):
        """
        Groups events into patches while maintaining original (global) x,y coordinates.
        """
        patch_h5_path = self.sequence_path / "events_patchified.h5"
        print(f"Creating patchified events for {self.sequence_id}...")

        # 1. Load raw events
        x = self.events['x'][:]
        y = self.events['y'][:]
        p = self.events['p'][:]
        t = self.events['t'][:]
        
        # 2. Assign patch IDs
        col = np.clip(x // self.patch_w, 0, self.n_w - 1).astype(np.int16)
        row = np.clip(y // self.patch_h, 0, self.n_h - 1).astype(np.int16)
        patch_ids = (row * self.n_w + col).astype(np.int16)

        num_events = len(x)
        global_to_local_idx = np.zeros(num_events, dtype=np.int32)

        with h5py.File(patch_h5_path, 'w') as f:
            for p_id in tqdm(range(self.num_patches), desc="Patchifying"):
                mask = (patch_ids == p_id)
                if not np.any(mask):
                    continue
                
                group = f.create_group(f"patch_{p_id}")
                
                # Save raw global coordinates
                group.create_dataset('x', data=x[mask], dtype=x.dtype, compression="gzip")
                group.create_dataset('y', data=y[mask], dtype=y.dtype, compression="gzip")
                group.create_dataset('p', data=p[mask], dtype=p.dtype, compression="gzip")
                group.create_dataset('t', data=t[mask], dtype=t.dtype, compression="gzip")
                
                # Map global index i to its position inside patch_p_id
                global_to_local_idx[mask] = np.arange(np.sum(mask), dtype=np.int32)

            f.create_dataset('global_to_local_idx', data=global_to_local_idx, compression="gzip")
            f.attrs['n_patches_h'] = self.n_h
            f.attrs['n_patches_w'] = self.n_w

    def __getitem__(self, index):
        """
        Loads a single S-sized chunk using a contiguous index.
        """
        patch_id, local_start = self.patch_offsets[index]
        group = self.patch_h5[f"patch_{patch_id}"]
        
        x = group['x'][local_start : local_start + self.S]
        y = group['y'][local_start : local_start + self.S]
        p = group['p'][local_start : local_start + self.S]
        t = group['t'][local_start : local_start + self.S]
        events_tensor =  torch.from_numpy(np.stack([x, y, p, t], axis=1).astype(np.float32))

        if self.rep_subsample_factor > 0:        
            rep = self.get_stacked_tsurface_representation(x=x, y=y, p=p, t=t)
        else:
            rep = self.get_time_surface(x=x, y=y, p=p, t=t) # (1, C, H, W)

        data = Data(
            x=events_tensor,
            sequence_id=self.sequence_id, 
            event_representation=rep 
        )
        return data

if __name__ == '__main__':
    dataset_path = Path("/users/rpellerito/scratch/datasets/DSEC/train")
    for seq_dir in dataset_path.iterdir():
        if seq_dir.is_dir():
            seq = IndexedPatchSequence(seq_path=seq_dir, 
                                       num_events=100000, 
                                       n_patches_h=16, 
                                       n_patches_w=16,
                                       rep_subsample_factor=10000
                                       )
            # Just initializing the sequence will generate the metadata if it doesn't exist
            item_ = seq[0]  # Accessing an item to ensure everything works