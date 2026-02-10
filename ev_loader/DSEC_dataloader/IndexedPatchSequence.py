import h5py
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from .Sequence import Sequence

class IndexedPatchSequence(Sequence):
    def __init__(self, 
                 n_patches_h: int = 4, 
                 n_patches_w: int = 4, 
                 min_events_per_patch: int = 512, 
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.S = min_events_per_patch
        self.n_h = n_patches_h
        self.n_w = n_patches_w
        self.num_patches = n_patches_h * n_patches_w
        
        self.patch_h = self.height // n_patches_h
        self.patch_w = self.width // n_patches_w
        
        # Access the raw event arrays from the base class slicer
        self.ev_raw = self.event_slicers["left"].events
        self.metadata_h5 = self.sequence_path / "patch_metadata.h5"
        
        if not self.metadata_h5.exists():
            self._generate_and_save_metadata()
        else:
            self.meta_f = h5py.File(str(self.metadata_h5), 'r')
            S = self.meta_f.attrs['S']
            n_h = self.meta_f.attrs['n_patches_h']
            n_w = self.meta_f.attrs['n_patches_w']
            if S != self.S or n_h != self.n_h or n_w != self.n_w:
                print(f"Metadata mismatch for {self.sequence_id}:")
                print(f"  Expected S={self.S}, n_h={self.n_h}, n_w={self.n_w}")
                print(f"  Found S={S}, n_h={n_h}, n_w={n_w}")
                print("Regenerating metadata with the correct parameters...")
                self.meta_f.close()
                self._generate_and_save_metadata()
        
        # Keep a reference to the metadata file
        self.meta_f = h5py.File(str(self.metadata_h5), 'r')
        self.patch_assignments = self.meta_f['patch_assignments']
        self.lookahead_indices = self.meta_f['lookahead_indices']

    def _generate_and_save_metadata(self):
        """
        Calculates for each event 'i', the index 'j' such that the window [i, j]
        contains AT LEAST S events for every patch. 
        This allows the loader to then extract EXACTLY the first S events 
        per patch within that range.
        """
        print(f"--- Generating H5 Metadata for {self.sequence_id} ---")
        
        x = self.ev_raw['x'][:]
        y = self.ev_raw['y'][:]
        col = np.clip(x // self.patch_w, 0, self.n_w - 1).astype(np.int16)
        row = np.clip(y // self.patch_h, 0, self.n_h - 1).astype(np.int16)
        patch_assignments = (row * self.n_w + col).astype(np.int16)

        num_events = len(patch_assignments)
        lookahead_indices = np.full(num_events, -1, dtype=np.int32)
        
        counts = np.zeros(self.num_patches, dtype=np.int32)
        idx_end = 0
        
        for idx_start in tqdm(range(num_events), desc="Forward Window Analysis"):
            # Move idx_end forward until every single patch counter is >= S
            while np.any(counts < self.S) and idx_end < num_events:
                p_id = patch_assignments[idx_end]
                counts[p_id] += 1
                idx_end += 1
            
            if np.all(counts >= self.S):
                # idx_end - 1 is the index of the event that finally 
                # satisfied the S requirement for the "slowest" patch.
                lookahead_indices[idx_start] = idx_end - 1
            else:
                # Cannot satisfy S for all patches anymore
                break
                
            # Pop the starting event from the counts to slide the window
            counts[patch_assignments[idx_start]] -= 1

            if idx_start > 1000000:
                break

        with h5py.File(self.metadata_h5, 'w') as f:
            f.create_dataset('patch_assignments', data=patch_assignments, compression="gzip")
            f.create_dataset('lookahead_indices', data=lookahead_indices, compression="gzip")
            f.attrs['S'] = self.S
            f.attrs['n_patches_h'] = self.n_h
            f.attrs['n_patches_w'] = self.n_w

    def __getitem__(self, index):
        start_idx = index
        end_idx = self.lookahead_indices[index]
        
        if end_idx == -1:
            # Fallback for the end of the sequence
            raise IndexError("Index out of range for the available events with the given S requirement.")

        # 1. Load the chunk of events for this window
        # We add 1 to end_idx because slicing is exclusive
        win_x = self.ev_raw['x'][start_idx : end_idx + 1]
        win_y = self.ev_raw['y'][start_idx : end_idx + 1]
        win_p = self.ev_raw['p'][start_idx : end_idx + 1]
        win_t = self.ev_raw['t'][start_idx : end_idx + 1]
        win_patches = self.patch_assignments[start_idx : end_idx + 1]

        # 2. Create the P x S x 4 Tensor
        # Shape: [Patches, Events_per_patch, Features(x,y,p,t)]
        output = np.zeros((self.num_patches, self.S, 4), dtype=np.float32)

        for p_id in range(self.num_patches):
            # Find indices where the event belongs to the current patch
            mask = (win_patches == p_id)
            
            # Extract exactly the first S events
            # (The metadata guarantees there are at least S available)
            px = win_x[mask][:self.S]
            py = win_y[mask][:self.S]
            pp = win_p[mask][:self.S]
            pt = win_t[mask][:self.S]

            # Localize coordinates
            c, r = p_id % self.n_w, p_id // self.n_w
            output[p_id, :, 0] = px - (c * self.patch_w)
            output[p_id, :, 1] = py - (r * self.patch_h)
            output[p_id, :, 2] = pp
            output[p_id, :, 3] = pt

        return torch.from_numpy(output)

    def __del__(self):
        if hasattr(self, 'meta_f'):
            self.meta_f.close()
            

if __name__ == '__main__':
    # Generate metadata for all sequences (only needs to be done once)
    dataset_path = Path("/users/rpellerito/scratch/datasets/DSEC/train")
    for seq_dir in dataset_path.iterdir():
        if seq_dir.is_dir():
            seq = IndexedPatchSequence(seq_path=seq_dir, min_events_per_patch=50000)
            # Just initializing the sequence will generate the metadata if it doesn't exist
            item_ = seq[0]  # Accessing an item to ensure everything works