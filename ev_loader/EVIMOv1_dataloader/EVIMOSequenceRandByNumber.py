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
from .EVIMOSequence import EVIMOSequence
from ev_loader.utils.utils_events_split import split_events


class EVIMOSequenceRandByNumber(EVIMOSequence):
    def __init__(self, h5_path: str, window_ms, num_bins, **kwargs):
        super().__init__(h5_path, window_ms, num_bins, **kwargs)
    
    # Loads events from final time which corresponds to GT
    # use the events from final index id1
    # Take a random number of events from that window, then compute delta t
    # Use the number of events in a windowd and take the same number preciding th
    def get_single_item(self, idx):
        # Load by index
        with torch.no_grad():
            id0 = self.start_ev_ind[idx]
            id1 = self.end_ev_ind[idx]
            
            n_events = id1 - id0
            #select a random number of events between a minim and a max number
            low = max(1, int(0.1 * n_events))
            high = max(low, int(0.9 * n_events))
            boundary_event_num = random.randint(low, high)
            
            id_mid = id1 - boundary_event_num
            # Assume the number of event in this window is representative 
            # of the time window "window_ms" we want to consider
            id_start = id_mid - n_events
            
            # clamp to the valid range of self.events
            id_start = max(id_start, 0)
            id_mid   = max(id_mid, 0)
            
            # If not enough events just take all the events
            short_window = (id_mid <= id_start) or n_events < 1000
            
            if short_window:
                events = self.events[id0:id1]
            else:
                events = self.events[id_start:id_mid]

            ev_x = events[:, 0]
            ev_y = events[:, 1]
            ev_t = events[:, 2]
            ev_p = events[:, 3]

            ev_x, ev_y, ev_p = self.event_augment(x=ev_x, y=ev_y, p=ev_p)
            voxel = self._events_to_voxel_grid(x=ev_x, y=ev_y, p=ev_p, t=ev_t)
            
            # Events for loss computation
            if short_window:
                # If not enough events just take all the events
                events = self.events[id0:id1]
            else:
                events = self.events[id_mid:id1]
                
            ev_x = events[:, 0]
            ev_y = events[:, 1]
            ev_t = events[:, 2]
            ev_p = events[:, 3]
            dt = ev_t[-1] - ev_t[0] # Expressed in seconds [s]
            
            if dt == 0:
                # force a tiny positive dt to avoid division by zero downstream
                # or resample a different window
                dt = 1e-6  # seconds, tiny but non-zero

            events_, pol_mask_, d_events, d_pol_mask = split_events(
                x_rect=ev_x,
                y_rect=ev_y,
                p=ev_p,
                t=ev_t, 
                max_num_grad_events=self.max_num_grad_events, 
                max_num_detach_events=self.max_num_detach_events
                )
            
            binary_mask = self.convert_to_binary_mask(self.mask[idx])

            return {
                "representation": voxel,
                "dynamic_mask": torch.from_numpy(binary_mask).unsqueeze(0),
                "sampled_dt": torch.tensor([dt]),
                "event_list": events_,
                "polarity_mask":pol_mask_,
                "d_event_list": d_events,
                "d_polarity_mask": d_pol_mask,
            }
    
    def _get_sequence_indices(self, index):
        """
        Load the sequence whose start is at sequence_len steps index and
        end at index. If index is not the first index, in this case we load 
        the sequence starting at index and ending at index + sequence_len +1.
        """
        # <= to avoid loading index 0
        if index <= self.sequence_len +1:
            # if the first index is negative use the forward sequence
            indices = list(range(index, index + self.sequence_len + 1))
            return indices
        start_ind = index - self.sequence_len - 1
        indices = list(range(start_ind, index))
        return indices
    
    def __getitem__(self, idx):
        indices = self._get_sequence_indices(idx)
        sequence_ = []
        for index in indices:
            data = self.get_single_item(index)
            sequence_.append(data)
        return sequence_
    
if __name__ == "__main__":
    # Example usage
    # h5_path = "/home/rpg/Downloads/EVIMO1/train/box/seq_00.h5"
    h5_path = "/data/scratch/pellerito/datasets/EVIMO1/train/box/seq_00.h5"


    dataset = EVIMOSequenceRandByNumber(h5_path, window_ms=100, num_bins=5)
    print(f"Number of samples: {len(dataset)}")
    
    for i in range(len(dataset)):
        sample = dataset[i]
        # print(f"Sample {i} sampled dt: {sample[0]['sampled_dt']}")
