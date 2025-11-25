"""
In this script we first select the ground truth then go back of a specified dt
finally from this timestamp we go back by 100ms and select the events.
This is to simulated as the gt that we have is dt in the future with respect to the events.
"""

from .HydraSequence import HydraSequence
from pathlib import Path
import numpy as np
import time
import torch

from evlicious import Events
import matplotlib.pyplot as plt

class HighFreqHydraSequence(HydraSequence):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def debug_plot(self, raw_events, dynamic_mask_t0):
        events = Events(
            x=raw_events['x'].astype(np.int16).astype(np.uint16),
            y=raw_events['y'].astype(np.int16).astype(np.uint16),
            t=raw_events['t'].astype(np.int64),
            p=raw_events['p'].astype(np.int8),
            width=640,
            height=480
        )
        rendered_events = events.render()

        plt.clf()
        plt.imshow(dynamic_mask_t0.squeeze().numpy())
        plt.imshow(rendered_events, alpha=0.5)
        plt.savefig('debug_plot.png')
        plt.close()

    
    def _get_single_datapoint_by_events(self, idx, events):
        dyn_mask_gt_t0_path = Path(self.dyn_masks_pathstrings[idx])
        dynamic_mask_t0 = self.get_dynamic_obj_mask(dyn_mask_gt_t0_path)
        dynamic_mask_t0 = torch.from_numpy(dynamic_mask_t0).float().unsqueeze(0)

        mask_for_out_events = (
            (events['x_0'] >= 0) & (events['x_0'] < 640) &
            (events['y_0'] >= 0) & (events['y_0'] < 480)
        )
            
        x_0 = events['x_0'][mask_for_out_events]
        y_0 = events['y_0'][mask_for_out_events]
        p_0 = events['p_0'][mask_for_out_events]
        t_0 = events['t_0'][mask_for_out_events]
        p_0 = p_0.astype(int)

        event_representation = self.get_event_representation(x=x_0, y=y_0, p=p_0, t=t_0)

        event_list, pol_mask, d_event_list, d_pol_mask = self.split_events(
            x_rect=x_0, y_rect=y_0, p=p_0, t=t_0
            )
        event_list = event_list.float()
        pol_mask = pol_mask.float()
        d_event_list = d_event_list.float()
        d_pol_mask = d_pol_mask.float()

        dynamic_mask_t0 = self.event_augment.augment_mask(dynamic_mask_t0)
        
        data = {
            'sequence_id': self.sequence_id,
            'representation': event_representation,
            'event_list': event_list, # N X 4
            'polarity_mask': pol_mask, # N X 2
            'd_event_list': d_event_list, # M X 4
            'd_polarity_mask': d_pol_mask, # M X 2
            'sampled_dt': self.dt/1000,
            'dynamic_mask': dynamic_mask_t0,
        }

        # self.debug_plot(
        #     raw_events={'x': x_0, 'y': y_0, 'p': p_0, 't': t_0},
        #     dynamic_mask_t0=dynamic_mask_t0
        # )
        return data
    
    def preload_events(self, indices, dt):
        t_s = self.timestamps[indices[0]] - self.delta_t_us - dt
        t_e = self.timestamps[indices[-1]]
        x, y, p, t = self.get_events_start_end_time(t_s, t_e)
        
        # rectify events with bilinear interpolation
        x, y = self.get_rectified_events(x, y)

        # Augment events
        x, y, p = self.event_augment(x=x, y=y, p=p)

        t_ends = self.timestamps[indices] - dt
        t_starts = t_ends - self.delta_t_us

        # Avoid going out of bounds
        t_ends[-1] = np.min((t_ends[-1], t.max()))

        events = []
        for t_start, t_end in zip(t_starts, t_ends):
            start_idx = np.argmin(np.abs(t - t_start))
            end_idx = np.argmin(np.abs(t - t_end))

            x_0 = x[start_idx:end_idx]
            y_0 = y[start_idx:end_idx]
            p_0 = p[start_idx:end_idx]
            t_0 = t[start_idx:end_idx]

            data = {'x_0': x_0, 'y_0': y_0, 'p_0': p_0, 't_0': t_0}
            events.append(data)
        return events
    
    def random_dt(self):
        """
        Randomly select a dt from the available dt options.
        """
        return np.random.choice(self.dt_ms, size=1, replace=False).item()

    def __getitem__(self, index):
        index += 1 if index == 0 else 0
        self.indices = self._get_sequence_indices(index)
        self.dt = self.random_dt()
        events = self.preload_events(indices=self.indices, dt=self.dt)

        sequence_data = []
        for i, idx in enumerate(self.indices):
            data = self._get_single_datapoint_by_events(idx=idx, events=events[i])
            sequence_data.append(data)

        return sequence_data

if __name__ == '__main__':
    # seq_abs_path = Path("//home/rpg/Downloads/DSEC/train/interlaken_00_c")
    # seq_abs_path = Path("/home/rpg/Downloads/DSEC/train/thun_00_a")
    seq_abs_path = Path("/data/scratch/pellerito/datasets/DSEC/train/interlaken_00_c")
    # dsec_seq = HydraSequence(
    #     seq_path=seq_abs_path, 
    #     num_bins=2, 
    #     sequence_len=2, 
    #     max_num_grad_events=10000, 
    #     dt=[1, 10, 100], 
    #     augment=["Horizontal", "Vertical", "Polarity"], 
    #     augment_prob=[0.5, 0.5, 0.5]
    #     )
    dsec_seq = HighFreqHydraSequence(
        seq_path=seq_abs_path, 
        num_bins=2, 
        sequence_len=2, 
        delta_t_ms=100,
        max_num_grad_events=10000, 
        dt=[1, 10, 100],
        augment=["Horizontal", "Vertical", "Polarity"], 
        augment_prob=[0.5, 0.5, 0.5]
        )

    for ind in range(len(dsec_seq)):
        loader_outputs = dsec_seq[ind]
        print(f"indices: {dsec_seq.indices}")
        print(f"index: {ind}")
        # if loader_outputs is None:
        #     print("None")
        #     continue
        for loader_output in loader_outputs:
            tm = time.time()
            sequence_id = loader_output['sequence_id']
            event_representation = loader_output['representation']
            dt = loader_output['sampled_dt']
            dynamic_mask = loader_output['dynamic_mask']

            event_list = loader_output['event_list']
            polarity_mask = loader_output['polarity_mask']
            d_event_list = loader_output['d_event_list']
            d_polarity_mask = loader_output['d_polarity_mask']
            
            print(f"-------------------------{(time.time()-tm)} dt {dt}-----------------------------")


