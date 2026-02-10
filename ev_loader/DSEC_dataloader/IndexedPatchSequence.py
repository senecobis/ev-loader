import h5py
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data

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
        
        self.generate_sequence()
        
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
    def generate_sequence(self):
        should_generate = False

        if not self.patch_h5_path.exists():
            should_generate = True
        else:
            try:
                # Open in read-only to check contents
                patch_ev = h5py.File(str(self.patch_h5_path), 'r')
                
                # 1. Check if required attributes exist
                if 'n_patches_h' not in patch_ev.attrs or 'n_patches_w' not in patch_ev.attrs:
                    print(f"Corrupted patchified H5 detected (missing attributes). Regenerating...")
                    should_generate = True
                else:
                    n_h = patch_ev.attrs['n_patches_h']
                    n_w = patch_ev.attrs['n_patches_w']
                    
                    # 2. Check if the grid matches current configuration
                    if n_h != self.n_h or n_w != self.n_w:
                        print(f"Grid mismatch ({n_h}x{n_w} vs {self.n_h}x{self.n_w}). Regenerating...")
                        should_generate = True
                
                # Always close before potential regeneration
                patch_ev.close()
                
            except (OSError, RuntimeError) as e:
                # This catches 'bad object header', 'file signature not found', etc.
                print(f"Cannot open {self.patch_h5_path.name} (likely corrupted): {e}")
                should_generate = True

        if should_generate:
            # If the file exists but is bad/wrong, remove it first to avoid 'w' mode issues
            if self.patch_h5_path.exists():
                self.patch_h5_path.unlink()
            self.generate_patchified_h5()

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

# --- Below is a utility script to generate patchified H5 files for all sequences in parallel ---
"""
Below is a utility script to generate patchified H5 files for all sequences in parallel 
Run as follows:
python -m ev_loader.DSEC_dataloader.IndexedPatchSequence
"""
from concurrent.futures import ProcessPoolExecutor
import functools

def process_single_sequence(seq_dir, n_h, n_w, s, rep_factor):
    """
    Worker function to initialize the sequence and trigger H5 generation.
    """
    try:
        print(f"--- Starting: {seq_dir.name} ---")
        # Initializing triggers generate_patchified_h5 if it doesn't exist
        seq = IndexedPatchSequence(
            seq_path=seq_dir, 
            num_events=s, 
            n_patches_h=n_h, 
            n_patches_w=n_w,
            rep_subsample_factor=rep_factor
        )
        
        # Accessing the first item ensures the H5 file is readable and offsets are correct
        _ = seq[0]
        
        # Explicitly close the file handle to prevent locks during parallel runs
        if hasattr(seq, 'patch_h5'):
            seq.patch_h5.close()
            
        print(f"--- Completed: {seq_dir.name} ---")
        return f"{seq_dir.name}: Success"
    except Exception as e:
        print(f"--- Error in {seq_dir.name}: {e} ---")
        return f"{seq_dir.name}: Failed"

def main():
    dataset_path = Path("/users/rpellerito/scratch/datasets/DSEC/train")
    
    # Parameters
    params = {
        'n_h': 16,
        'n_w': 16,
        's': 100000,
        'rep_factor': 10000
    }

    # Gather all sequence directories
    seq_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    # Determine number of workers. 
    # Warning: Each worker loads the whole sequence events into RAM.
    # DSEC sequences can be large (~5-10GB each). 
    # If you have 128GB RAM, you can probably use 8-10 workers safely.
    # max_workers = min(len(seq_dirs), 40) # Adjust based on your RAM
    max_workers = min(len(seq_dirs), 12) # Adjust based on your RAM


    print(f"Starting parallel patchification with {max_workers} workers...")

    # Use a partial function to pass our constant parameters
    worker_func = functools.partial(
        process_single_sequence, 
        n_h=params['n_h'], 
        n_w=params['n_w'], 
        s=params['s'], 
        rep_factor=params['rep_factor']
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(worker_func, seq_dirs))

    print("\nSummary of results:")
    for res in results:
        print(res)

if __name__ == '__main__':
    main()


# if __name__ == '__main__':

#     seq_dir = Path("/iopsstor/scratch/cscs/rpellerito/datasets/DSEC/train/thun_00_a")

#     seq = IndexedPatchSequence(seq_path=seq_dir,
#         num_events=100000,
#         n_patches_h=16,
#         n_patches_w=16,
#         rep_subsample_factor=10000
#         )
#     item_ = seq[0] # Accessing an item to ensure everything works