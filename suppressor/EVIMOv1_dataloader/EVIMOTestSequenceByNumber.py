"""
Script to load EVIMO v1 dataset for training by Roberto Pellerito rpellerito@ifi.uzh.ch
This dataloader randomly selects a number of events within a time window
to form the input representation, and uses the following events for loss computation.

Most importantly to avoid loading by time which is computationally expensive,
we load by number of events.
We use an event number boundary to split the events in the window before the GT
and the events after the GT for loss computation.

We consider a number of events equivalent to a time window.
"""

import random
import torch
import numpy as np
from .EVIMOSequence import EVIMOSequence
from suppressor.utils.utils_events_split import split_events


class EVIMOTestSequenceByNumber(EVIMOSequence):
    def __init__(self, h5_path: str, window_ms: float, num_bins: int, dt_pred_time_ms: float, **kwargs):
        super().__init__(h5_path, window_ms, num_bins, **kwargs)
        self.dt_pred_time = dt_pred_time_ms / 1000.0  # prediction time in seconds [s]

        events_t = self.h5_file['events_t']
        total_window = self.window + self.dt_pred_time
        
        start_timestamps = self.timestamps[()] - total_window
        start_timestamps = np.clip(start_timestamps, a_min=0, a_max=None)
        
        mid_timestamps = self.timestamps[()] - self.dt_pred_time
        mid_timestamps = np.clip(mid_timestamps, a_min=0, a_max=None)

        self.start_ev_ind = np.searchsorted(events_t, start_timestamps)
        self.mid_ev_ind = np.searchsorted(events_t, mid_timestamps)
        # We can avoid searching for the end indices as we don't load the events after midpoint

    def get_single_item(self, idx):
        # Load by index
        with torch.no_grad():
            id0 = self.start_ev_ind[idx]
            id1 = self.mid_ev_ind[idx] # mid index corresponds to the end of the input events
            
            events = self.events[id0:id1]

            ev_x = events[:, 0]
            ev_y = events[:, 1]
            ev_t = events[:, 2]
            ev_p = events[:, 3]

            voxel = self._events_to_voxel_grid(x=ev_x, y=ev_y, p=ev_p, t=ev_t)
                        
            binary_mask = self.convert_to_binary_mask(self.mask[idx])

            return {
                "events": events,
                "representation": voxel,
                "dynamic_mask": torch.from_numpy(binary_mask).unsqueeze(0).float(),
                "sampled_dt": torch.tensor([self.dt_pred_time]), # is in seconds [s]
                # For EV-IMO paper metrics computation
                "sequence_id": self.sequence_id,
                "depth_map": torch.from_numpy(self.depth[idx])
            }

    def __getitem__(self, idx):
        if self.start_ev_ind[idx] == self.mid_ev_ind[idx]:
            idx = idx + 1
            return self.__getitem__(idx)
        else:
            return self.get_single_item(idx)
    
if __name__ == "__main__":
    # Example usage
    h5_path = "/data/scratch/pellerito/datasets/EVIMO1/train/box/seq_00.h5"

    dataset = EVIMOTestSequenceByNumber(h5_path, window_ms=100, num_bins=5, dt_pred_time_ms=50)
    print(f"Number of samples: {len(dataset)}")
    
    for i in range(len(dataset)):
        
        sample = dataset[i]
        print(f"Sample {i} sampled dt: {sample['sampled_dt']}")
        time_difference = sample['events'][-1,2] - sample['events'][0,2]
        print(f"Sample {i} time difference in events: {time_difference}")
