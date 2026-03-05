"""Class for loading Gen1 data as patches of events from Roberto Pellerito (rpellerito@ifi.uzh.ch). 
    This is a more complex dataloader that processes raw event data into patches 
    and stores them in a way that allows for efficient loading during training.
    
    Stores the bounding boxes as [class_id, x_tl, y_tl, w, h]


Returns:
    _type_: _description_
"""

import os
import numpy as np
import pickle
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
from typing import List

from .utils import normalize_time
from .sequence import Gen1, PSEELoader
from ..utils.VectorizedPatchfier import VectorizedPatchfier



class PatchedGen1(Gen1):
    def __init__(self, n_patches_h=16, n_patches_w=16, **kwargs):
        # Pass parameters to the original Gen1
        super().__init__(**kwargs)
        self.n_h = n_patches_h
        self.n_w = n_patches_w
        
        # We initialize the patchfier here for use in processing
        self.patchfier = VectorizedPatchfier(
            input_H=240,
            input_W=304,
            patch_grid_H=self.n_h,
            patch_grid_W=self.n_w
        )
    
    @property
    def root(self) -> str:
        return os.path.join(os.environ["GEN1_DATA_DIR"], "gen1")

    def _processing(self, rf, root, read_annotations, read_label, buffer_to_data):
        data_loader = PSEELoader(rf)
        bounding_boxes = read_annotations(rf)
        labels = np.array(read_label(bounding_boxes))

        for i, bbox in enumerate(bounding_boxes):
            processed_dir = os.path.join(root, "processed")
            processed_file = rf.replace(root, processed_dir).replace(".dat", f"{i}.pkl")
            if os.path.exists(processed_file):
                continue

            # Determine temporal window around the current bounding box [t_start, t_end]. Add all of the
            # bounding boxes within this window to the sample.
            sample_dict = dict()
            sample_dict['patch_h'] = self.n_h
            sample_dict['patch_w'] = self.n_w

            t_bbox = bbox[0]
            t_start = t_bbox - 100000  # 100 ms
            t_end = t_bbox + 300000  # 300 ms
            bbox_mask = np.logical_and(t_start < bounding_boxes['ts'], bounding_boxes['ts'] < t_end)

            sample_dict['bbox'] = torch.tensor(bounding_boxes[bbox_mask].tolist())
            sample_dict['label'] = labels[bbox_mask]
            sample_dict['raw_file'] = rf

            # Load raw data around bounding box.
            idx_start = data_loader.seek_time(t_start)
            data = data_loader.load_delta_t(t_end - t_start)
            sample_dict['raw'] = (idx_start, data.size)  # offset and number of events
            if data.size < 4000:
                continue
        
            # 1. Convert loaded buffer to torch for patching
            # Assume 'data' is the output from data_loader.load_delta_t
            ev_tensor = torch.from_numpy(np.stack([
                data['x'], data['y'], data['t'], data['p']
            ], axis=1)).float().unsqueeze(0) # (1, N, 4)

            # 2. Patchify (Reorders events by patch and makes coords local)
            patched_events, mask = self.patchfier(ev_tensor, to_local_coords=True)
            
            # B=Batch, P=Patches, S=Max events per patch, C=Channels
            B, P, S, C = patched_events.shape
            x_long = patched_events.reshape(-1, C)
            mask_long = mask.reshape(-1)
            
            # Create patch IDs for every event index
            patch_ids = torch.arange(P).repeat_interleave(S)

            # 3. Filter only valid events (remove padding)
            x_long = x_long[mask_long]
            patch_ids = patch_ids[mask_long]

            # 4. Calculate cu_seqlens
            counts = torch.bincount(patch_ids, minlength=P)
            cu_seqlens = torch.zeros(counts.shape[0] + 1, dtype=torch.int32)
            torch.cumsum(counts, dim=0, out=cu_seqlens[1:])

            # Save this to your pickle
            sample_dict['x_patched'] = x_long
            sample_dict['patch_ids'] = patch_ids
            sample_dict['cu_seqlens'] = cu_seqlens
            
            # Store resulting dictionary in file
            os.makedirs(os.path.dirname(processed_file), exist_ok=True)
            with open(processed_file, 'wb') as f:
                pickle.dump(sample_dict, f)
            
            
    def _load_processed_file(self, f_path: str) -> Data:
        """ 
        This name MUST match exactly to override Gen1._load_processed_file
        """
        with open(f_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        patch_w = data_dict['patch_w']
        patch_h = data_dict['patch_h']
        assert patch_w == self.n_w and patch_h == self.n_h, \
            "Patch dimensions in file do not match the initialized dimensions of the dataloader."
        
        # Construct the Data object using the keys we saved in _processing
        data = Data(
            x=data_dict['x_patched'],
            patch_ids=data_dict['patch_ids'],
            cu_seqlens=data_dict['cu_seqlens']
        )
        
        # Handle the annotations/labels
        data.bbox = data_dict['bbox'][:, 1:6].long()
        data.y = data.bbox[:, -1]
        data.label = data_dict['label']
        
        # Normalize time (assuming index 2 is time: x, y, t, p)
        data.x[:, 2] = normalize_time(data.x[:, 2])
        
        return data
            

    @staticmethod
    def collate_fn(data_list: List[Data]) -> Batch:
        # 1. Use the default PyG batching to concatenate everything into one "Long" graph
        # This automatically creates batch.x, batch.batch, batch.ptr, etc.
        batch = Batch.from_data_list(data_list)
        
        # 2. Convert the concatenated 'x' to a dense tensor (B, Max_S, C)
        # x_dense: [Batch, Max_Events, 4]
        # mask: [Batch, Max_Events] (True for real events, False for padding)
        x_dense, mask = to_dense_batch(batch.x, batch.batch)
        
        # 3. Convert 'patch_ids' to dense as well
        # Note: patch_ids are integers, so they will be padded with 0 by default. 
        # If 0 is a valid patch ID, you might want to fill padding with -1 later.
        patch_ids_dense, _ = to_dense_batch(batch.patch_ids, batch.batch, fill_value=-1)

        # 4. Handle cu_seqlens
        # Since n_patches is fixed, we can just stack them.
        # cu_seqlens are already relative to the start of each sample (0), 
        # so stacking them is correct for a dense batch.
        cu_seqlens_dense = torch.stack([d.cu_seqlens for d in data_list], dim=0)

        # Re-assign the dense versions to the batch object
        batch.x = x_dense
        batch.patch_ids = patch_ids_dense
        batch.cu_seqlens = cu_seqlens_dense
        batch.mask = mask # Standard name for padding masks

        # Keep original bounding box batch mapping
        if hasattr(data_list[0], 'bbox'):
            batch_bbox = sum([[i] * len(data.y) for i, data in enumerate(data_list)], [])
            batch.batch_bbox = torch.tensor(batch_bbox, dtype=torch.long)
            
        # the original bounding boxes have the format (x, y, w, h, class_id)
        # transform it to (class_id, x, y, w, h) for YOLOX format, reshuffle the columns
        batch.bbox = batch.bbox[:, [4, 0, 1, 2, 3]]
        
            
        return batch