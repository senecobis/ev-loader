import os
import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Data

from .TimeSurfaceSequence import TimeSurfaceSequence
from ..utils.VectorizedPatchfier import VectorizedPatchfier

from evlicious import Events
import matplotlib.pyplot as plt
from ..representations.time_surface import ToTimesurface

class PatchedTimeSurfaceSequence(TimeSurfaceSequence):
    def __init__(self, 
                 n_patches_h=16, 
                 n_patches_w=16, 
                 num_events=10000, 
                 rep_subsample_factor: int = -1,
                 debug=False, 
                 **kwargs):
        super().__init__(num_events=num_events, 
                         rep_subsample_factor=rep_subsample_factor, 
                         **kwargs)
        self.debug = debug
        self.n_h = n_patches_h
        self.n_w = n_patches_w
        self.num_patches = n_patches_h * n_patches_w
        self.patch_h = self.height // n_patches_h
        self.patch_w = self.width // n_patches_w

        self.patchfier = VectorizedPatchfier(
            input_H=self.height,
            input_W=self.width,
            patch_grid_H=self.n_h,
            patch_grid_W=self.n_w
        )
        self.events = self.event_slicers["left"].events

        self.to_timesurface = ToTimesurface(sensor_size=(self.width, self.height, 2), 
                                    surface_dimensions=None, 
                                    tau=1.0, 
                                    decay="exp"
                                    )


    def normalise_coordinates(self, x, y, patch_id):
        # col = patch index in horizontal direction, row = vertical
        col = patch_id % self.n_w
        row = patch_id // self.n_w

        x_offset = col * self.patch_w
        y_offset = row * self.patch_h

        x_local = x - x_offset
        y_local = y - y_offset
        return x_local, y_local
    
    def normalise_timestamps(self, t):
        t_norm = t - t[0]
        if t_norm[-1] > 0:
            t_norm = t_norm / t_norm[-1]
        return t_norm

    def visualise_patch(self, x, y, p, t):
        t = t.astype(np.int64)
        p = p.astype(np.int8)
        ev = Events(x=x, y=y, p=p, t=t, width=self.patch_w, height=self.patch_h)
        rendered = ev.render()
        plt.imshow(rendered)
        plt.title(f"Visualisation of patch events (N={len(ev)})")
        os.makedirs("debug_viz", exist_ok=True)

        plt.savefig(f"debug_viz/patch_viz_{self.sequence_id}.png")

    def visualise_time_surface(self, rep):
        rep = rep.squeeze(0).numpy()  # (C, H, W)
        plt.imshow(rep[0], cmap='hot')  # Visualize the first channel of the time surface
        plt.title(f"Time Surface of sequence {self.sequence_id}")
        os.makedirs("debug_viz", exist_ok=True)
        plt.savefig(f"debug_viz/ts_viz_{self.sequence_id}.png")

    def visualise_stitched_patches(self, patched_events, mask):
        B, P, S, C = patched_events.shape
        assert B == 1, "Batch size > 1 not supported for visualization"
        
        stitched_image = np.zeros((self.height, self.width), dtype=np.uint8)
        
        for patch_id in range(P):
            valid_events = patched_events[0, patch_id][mask[0, patch_id]]  # (Num_Valid, C)
            if valid_events.shape[0] == 0:
                continue
            
            x_local = valid_events[:, 0].numpy().astype(np.uint16)
            y_local = valid_events[:, 1].numpy().astype(np.uint16)
            
            col = patch_id % self.n_w
            row = patch_id // self.n_w
            x_offset = col * self.patch_w
            y_offset = row * self.patch_h
            
            x_global = x_local + x_offset
            y_global = y_local + y_offset
            
            stitched_image[y_global, x_global] = 255  # Mark event locations
        
        plt.imshow(stitched_image, cmap='gray')
        plt.title(f"Stitched Patches of sequence {self.sequence_id}")
        os.makedirs("debug_viz", exist_ok=True)
        plt.savefig(f"debug_viz/stitched_patches_{self.sequence_id}.png")
        
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

        t_norm = self.normalise_timestamps(t)
        rep = self.get_time_surface(x=x, y=y, p=p, t=t_norm) # (1, C, H, W)
        
        events = torch.from_numpy(np.stack([x, y, t_norm, p], axis=1)).float().unsqueeze(0) # (1, N, 4)
        patched_events, mask = self.patchfier(events, 
                                              max_seq_len=None, 
                                              to_local_coords=True
                                              )
        if self.debug:
            # Visualise the first patch as a sanity check
            first_patch_events = patched_events[0, 0]  # (S, C)
            valid_mask = mask[0, 0]  # (S,)
            valid_events = first_patch_events[valid_mask]  # (Num_Valid, C)
            if valid_events.shape[0] > 0:
                x_vis = valid_events[:, 0].numpy().astype(np.uint16)
                y_vis = valid_events[:, 1].numpy().astype(np.uint16)
                p_vis = valid_events[:, 3].numpy()  # Polarity is at index 3
                t_vis = valid_events[:, 2].numpy()  # Timestamp is at index 2
                self.visualise_patch(x_vis, y_vis, p_vis, t_vis)
            self.visualise_stitched_patches(patched_events, mask)
            self.visualise_time_surface(rep)

        B, P, S, C = patched_events.shape
        x_long = patched_events.reshape(-1, C)  # Shape: (P*S, 4)
        mask_long = mask.reshape(-1)            # Shape: (P*S)
        patch_ids = torch.arange(P, dtype=torch.long).repeat_interleave(S) # Shape: (P*S)

        x_long = x_long[mask_long]  # Keep only valid events
        patch_ids = patch_ids[mask_long]  # Corresponding patch IDs for valid events

        counts = torch.bincount(patch_ids)
        cu_seqlens = torch.zeros(counts.shape[0] + 1, device=x_long.device, dtype=torch.int32)
        torch.cumsum(counts, dim=0, out=cu_seqlens[1:])

        data = Data(
            x=x_long.unsqueeze(0),
            sequence_id=self.sequence_id, 
            event_representation=rep,
            cu_seqlens=cu_seqlens,
            patch_ids=patch_ids
        )

        return data



if __name__ == '__main__':

    seq_dir = Path("/iopsstor/scratch/cscs/rpellerito/datasets/DSEC/test/interlaken_00_a")
    
    seq = PatchedTimeSurfaceSequence(seq_path=seq_dir,
        num_events=1000000,
        n_patches_h=16,
        n_patches_w=16,
        rep_subsample_factor=0,
        debug=True
        )
    for item_ in seq:
        print(item_)
