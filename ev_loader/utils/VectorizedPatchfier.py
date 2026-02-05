import torch
import torch.nn as nn
import time

class VectorizedPatchfier(nn.Module):
    def __init__(
        self,
        input_H=256,
        input_W=256,
        patch_grid_H=2,
        patch_grid_W=2,
    ):
        super().__init__()
        self.input_H = input_H
        self.input_W = input_W
        self.patch_grid_H = patch_grid_H
        self.patch_grid_W = patch_grid_W
        self.num_patches = patch_grid_H * patch_grid_W
        
        self.patch_size_h = input_H // patch_grid_H
        self.patch_size_w = input_W // patch_grid_W
        
        # Pre-compute offsets for local coordinate transformation
        y_ids = torch.arange(patch_grid_H).repeat_interleave(patch_grid_W)
        x_ids = torch.arange(patch_grid_W).repeat(patch_grid_H)
        
        self.register_buffer('patch_offsets_x', x_ids * self.patch_size_w)
        self.register_buffer('patch_offsets_y', y_ids * self.patch_size_h)

    def forward(self, events, max_seq_len=128, to_local_coords=True):
        """
        Args:
            events: (B, S, C) - Example: (1, 233491, 4)
                    Assumes channel C last. x is index 0, y is index 1.
            max_seq_len: int
            to_local_coords: bool
        Returns:
            out_events: (B, Num_Patches, max_seq_len, C)
            out_mask:   (B, Num_Patches, max_seq_len)
        """
        # Shape is now (Batch, Sequence, Channel)
        B, S, C = events.shape
        device = events.device
        
        # 1. Flatten Batch and Sequence: (B*S, C)
        flat_events = events.reshape(-1, C)
        
        # We assume x is index 0 and y is index 1
        x_flat = flat_events[:, 0]
        y_flat = flat_events[:, 1]
        
        # 2. Calculate Patch IDs (Vectorized)
        patch_x = (x_flat // self.patch_size_w).clamp(0, self.patch_grid_W - 1)
        patch_y = (y_flat // self.patch_size_h).clamp(0, self.patch_grid_H - 1)
        patch_ids = (patch_y * self.patch_grid_W + patch_x).long()
        
        # 3. Create a unique "Group ID" for every (Batch, Patch) pair
        # Batch indices: [0, 0... 0, 1, 1... 1, ...]
        batch_indices = torch.arange(B, device=device).repeat_interleave(S)
        
        # Group ID = Batch_Index * Num_Patches + Patch_ID
        group_ids = batch_indices * self.num_patches + patch_ids
        
        # 4. Sort by Group ID
        # stable=True maintains temporal order
        sorted_group_ids, sort_indices = torch.sort(group_ids, stable=True)
        
        # 5. Calculate "Rank" within each group
        unique_groups, counts = torch.unique_consecutive(sorted_group_ids, return_counts=True)
        
        ends = counts.cumsum(dim=0)
        starts = torch.cat([torch.zeros(1, device=device, dtype=torch.long), ends[:-1]])
        repeated_starts = torch.repeat_interleave(starts, counts)
        
        ranks = torch.arange(sorted_group_ids.shape[0], device=device) - repeated_starts
        
        # 6. Filter out ranks that exceed max_seq_len
        valid_mask = ranks < max_seq_len
        
        final_src_indices = sort_indices[valid_mask]
        final_group_ids = sorted_group_ids[valid_mask] 
        final_ranks = ranks[valid_mask]                
        
        # 7. Scatter into Output Tensor
        # Output shape: (B * P, MaxLen, C)
        out_events = torch.zeros(B * self.num_patches, 
                                 max_seq_len, 
                                 C, 
                                 device=device,
                                 dtype=events.dtype)
        
        out_mask = torch.zeros(B * self.num_patches, max_seq_len, dtype=torch.bool, device=device)
        
        # Retrieve valid events
        valid_events = flat_events[final_src_indices] # (Num_Valid, C)
        
        if to_local_coords:
            relevant_patch_ids = final_group_ids % self.num_patches
            off_x = self.patch_offsets_x[relevant_patch_ids]
            off_y = self.patch_offsets_y[relevant_patch_ids]
            
            valid_events[:, 0] -= off_x
            valid_events[:, 1] -= off_y

        # Scatter
        # Dest Index = Group_ID * Max_Len + Rank
        dest_indices = final_group_ids * max_seq_len + final_ranks
        
        # Flatten the first two dims of output for scatter assignment
        # out_events view: (B*P*MaxLen, C)
        out_events.view(-1, C)[dest_indices] = valid_events
        out_mask.view(-1)[dest_indices] = True
        
        # 8. Final Reshape to (B, P, S, C)
        out_events = out_events.view(B, self.num_patches, max_seq_len, C)
        out_mask = out_mask.view(B, self.num_patches, max_seq_len)
        
        return out_events, out_mask

# --- Verification ---
if __name__ == "__main__":
    # Setup for your specific shape
    B, S, C = 1, 233491, 4
    H, W = 256, 256
    
    # Create random events with shape (B, S, C)
    events = torch.zeros(B, S, C)
    events[:, :, 0] = torch.randint(0, W, (B, S))       # x
    events[:, :, 1] = torch.randint(0, H, (B, S))       # y
    events[:, :, 2] = torch.arange(S).float().repeat(B, 1) # t
    events[:, :, 3] = torch.randint(0, 2, (B, S))       # p
    
    model = VectorizedPatchfier(input_H=H, input_W=W, patch_grid_H=4, patch_grid_W=4)
    
    if torch.cuda.is_available():
        model = model.cuda()
        events = events.cuda()
    
    start = time.time()
    out_ev, out_mask = model(events, max_seq_len=1024, to_local_coords=True)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    end = time.time()

    print(f"Input Shape:  {events.shape}")
    print(f"Output Shape: {out_ev.shape}") # Expected: (1, 16, 1024, 4)
    print(f"Mask Shape:   {out_mask.shape}")
    print(f"Time taken:   {end - start:.4f}s")