"""
Script to load EVIMO v1 dataset for training by Roberto Pellerito rpellerito@ifi.uzh.ch
"""

import torch
from .EVIMOSequence import EVIMOSequence

class EVIMOTestSequenceImportByNumber(EVIMOSequence):
    def __init__(self, h5_path: str, window_ms, num_bins, representation="voxel", window_ev_num=None):
        super().__init__(h5_path, window_ms, num_bins)
        id_ = self.h5_path.split("/")[-1].split(".")[0]
        scene_ = self.h5_path.split("/")[-2]
        self.sequence_id = f"{scene_}_{id_}"
        
        self.representations = ['voxel', 'oms']
        assert representation in self.representations, f"Representation {representation} not supported. Choose from {self.representations}"
        self.representation = representation

        self.window_ev_num = window_ev_num 
        
    def get_representation(self, x, y, t, p):
        if self.representation == 'voxel':
            voxel = self._events_to_voxel_grid(x=x, y=y, p=p, t=t)
            return voxel
        elif self.representation == 'oms':
            p[p==0] = 1.0
            return self._events_to_voxel_grid(x=x, y=y, p=p, t=t)
        else:
            raise ValueError(f"Representation {self.representation} not supported.")
    
    def get_single_item(self, idx):
        with torch.no_grad():
            id0 = self.start_ev_ind[idx]
            id1 = self.end_ev_ind[idx]

            if self.window_ev_num is not None:
                id0 = id1 - self.window_ev_num
            
            if id0 < 0:
                id0 = 0

            events = self.events[id0:id1]
            ev_x = events[:, 0]
            ev_y = events[:, 1]
            ev_t = events[:, 2]
            ev_p = events[:, 3]

            dt = ev_t[-1] - ev_t[0]
            representation = self.get_representation(x=ev_x, y=ev_y, p=ev_p, t=ev_t)
            
            binary_mask = self.convert_to_binary_mask(self.mask[idx])

            return {
                "events": events,
                "representation": representation,
                "mask": torch.from_numpy(self.mask[idx]),
                "dynamic_mask": torch.from_numpy(binary_mask).unsqueeze(0).float(),
                "sampled_dt": torch.tensor([dt], dtype=torch.float32, requires_grad=False),
                "sequence_id": self.sequence_id
            }
    
    def __getitem__(self, idx):        
        return self.get_single_item(idx)
    