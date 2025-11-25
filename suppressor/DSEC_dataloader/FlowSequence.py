from ..utils.representations import create_cnt_encoding
from .sequence import Sequence
from pathlib import Path
import numpy as np
import torch


class FlowSequence(Sequence):
    def __init__(self, sequence_len:int = 2, max_num_grad_events:int = 10000, dt:list = [0.0], **kwargs):
        super().__init__(**kwargs)
        self.sequence_len = sequence_len
        self.max_num_grad_events = max_num_grad_events # max num of event to be used for backprop

        # NOTE DSEC event timestamps are in us (micro seconds)
        # dt_ms is written in milliseconds, so we multiply by 1000 to convert to microseconds
        # e.g. dt = [1, 10, 100] means future time in milliseconds [ms] e.g. 1 ms, 10 ms, 100 ms
        self.dt_ms = np.array(dt) * 1000

    @staticmethod
    def create_polarity_mask(ps):
        """
        Creates a two channel tensor that acts as a mask for the input event list.
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [N x 2] polarity list event representation
        """
        ps = torch.from_numpy(ps)
        event_list_pol_mask = torch.stack([ps, ps])
        event_list_pol_mask[0, :][event_list_pol_mask[0, :] < 0] = 0
        event_list_pol_mask[0, :][event_list_pol_mask[0, :] > 0] = 1
        event_list_pol_mask[1, :][event_list_pol_mask[1, :] < 0] = -1
        event_list_pol_mask[1, :][event_list_pol_mask[1, :] > 0] = 0
        event_list_pol_mask[1, :] *= -1
        return event_list_pol_mask
    
    @staticmethod
    def split_event_list(event_list, event_list_pol_mask, max_num_grad_events):
        """
        Splits the event list into two lists, one of them (with max. length) to be used for backprop.
        This helps reducing (VRAM) memory consumption.
        :param event_list: [4 x N] list event representation
        :param event_list_pol_mask: [2 x N] polarity list event representation
        :param max_num_grad_events: maximum number of events to be used for backprop
        :return event_list: [4 x N] list event representation to be used for backprop
        :return event_list_pol_mask: [2 x N] polarity list event representation to be used for backprop
        :return d_event_list: [4 x N] list event representation
        :return d_event_list_pol_mask: [2 x N] polarity list event representation
        """

        d_event_list = torch.zeros((4, 0))
        d_event_list_pol_mask = torch.zeros((2, 0))
        if max_num_grad_events is not None and event_list.shape[1] > max_num_grad_events:
            probs = torch.ones(event_list.shape[1], dtype=torch.float32) / event_list.shape[1]
            sampled_indices = probs.multinomial(
                max_num_grad_events, replacement=False
            )  # sample indices with equal prob.

            unsampled_indices = torch.ones(event_list.shape[1], dtype=torch.bool)
            unsampled_indices[sampled_indices] = False
            d_event_list = event_list[:, unsampled_indices]
            d_event_list_pol_mask = event_list_pol_mask[:, unsampled_indices]

            event_list = event_list[:, sampled_indices]
            event_list_pol_mask = event_list_pol_mask[:, sampled_indices]

        return event_list, event_list_pol_mask, d_event_list, d_event_list_pol_mask

    @staticmethod
    def norm_p(p):
        p = p.astype(np.int8)
        p[p < 1] = -1
        return p
    
    @staticmethod
    def norm_t(t):
        t = (t - t[0]) / (t[-1] - t[0])
        return t
    
    def get_cnt(self, x_, y_, p_):
        p_ = self.norm_p(p_)
        rect_mapping=self.rectify_ev_maps["left"]
        sensor_size=(self.height, self.width)
        x_ = torch.from_numpy(x_).float()
        y_ = torch.from_numpy(y_).float()
        p_ = torch.from_numpy(p_).float()
        event_representation = create_cnt_encoding(x_, y_, p_, rect_mapping, sensor_size, device='cpu')
        return event_representation
        
    def get_events_and_polarity_mask(self, x, y, p, t):
        x_rect, y_rect = self.get_rectified_events(x, y)
        p = self.norm_p(p)
        t = self.norm_t(t)

        # Split the event list into two lists, one of them 
        # (with max. length) to be used for backprop the other just for loss computation
        event_list_pol_mask = self.create_polarity_mask(p)
        raw_events = torch.from_numpy(np.stack([t, y_rect, x_rect, p], axis=0))
        # Split events for backprop and events just for calculation (detached)
        event_list, polarity_mask, d_event_list, d_polarity_mask = self.split_event_list(
            raw_events, event_list_pol_mask, self.max_num_grad_events
        )

        # Flip to have the same convention of the Iterative Warping loss
        event_list = event_list.permute(1, 0)
        polarity_mask = polarity_mask.permute(1, 0)
        d_event_list = d_event_list.permute(1, 0)
        d_polarity_mask = d_polarity_mask.permute(1, 0)
        return event_list, polarity_mask, d_event_list, d_polarity_mask
    
        
    def __getitem__(self, index):
        if index < self.sequence_len-1:
            # the actual index is smaller than the sequence length so we cannot create a sequence
            # we will return the first valid sequence, hence the index would be moved forward
            index = self.sequence_len-1
        # if the actual index is smaller than the sequence length should be handled here in indices
        indices = list(range(max(0, index - self.sequence_len + 1), index + 1))

        # TODO This function load events twice for future events, realoding events might be slow
        sequence_data = []
        for idx in indices:
            disp_gt_path = Path(self.disp_gt_pathstrings[idx])
            file_index = int(disp_gt_path.stem)
            
            # Just move the end tstamp forward in time
            # --> Create the cnt encoding with events at current t_stamp
            sampled_dt = np.random.choice(self.dt_ms, size=1, replace=False).item()
            
            ts_end = self.timestamps[idx]
            ts_mid_point = ts_end - sampled_dt
            ts_start = ts_mid_point - self.delta_t_us
            # |ts_start----ts_mid_point----ts_end|
            
            x,y,p,t = self.get_events_start_end_time(ts_start, ts_end)
            time_differences = np.abs(t - ts_mid_point)
            closest_index = np.argmin(time_differences)
            x_0 = x[:closest_index]
            y_0 = y[:closest_index]
            p_0 = p[:closest_index]

            # TODO add events augmentation
            # Create event count representation and rectify events with bilinear interpolation
            event_representation = self.get_cnt(x_=x_0, y_=y_0, p_=p_0)

            # --> Load future events for training future optical flow
            x_1 = x[closest_index:]
            y_1 = y[closest_index:]
            p_1 = p[closest_index:]
            t_1 = t[closest_index:]
            event_list, polarity_mask, d_event_list, d_polarity_mask = self.get_events_and_polarity_mask(x=x_1, y=y_1, p=p_1, t=t_1)

            data = {
                'file_index': file_index,
                'sequence_id': self.sequence_id,
                'intrinsics': self.K,
                'representation': {"left": event_representation},
                'event_list': event_list, # N X 4
                'polarity_mask': polarity_mask, # N X 2
                'd_event_list': d_event_list, # N X 4
                'd_polarity_mask': d_polarity_mask, # N X 2
                'sampled_dt': sampled_dt/1000
            }    
            sequence_data.append(data)

        return sequence_data


if __name__ == '__main__':
    # seq_abs_path = Path("//home/rpg/Downloads/DSEC/train/interlaken_00_c")
    # seq_abs_path = Path("/home/rpg/Downloads/DSEC/train/thun_00_a")
    seq_abs_path = Path("/data/scratch/pellerito/datasets/DSEC/train/thun_00_a")
    dsec_seq = FlowSequence(
        seq_path=seq_abs_path, num_bins=2, sequence_len=2, max_num_grad_events=10000, dt=[1, 10, 100]
        )
    loader_outputs = dsec_seq[0]

    for loader_output in loader_outputs:
        file_index = loader_output['file_index']
        sequence_id = loader_output['sequence_id']
        intrinsics = loader_output['intrinsics']
        event_representation = loader_output['representation']["left"]
        polarity_mask = loader_output['polarity_mask']
        raw_events = loader_output['event_list']
        d_pol_mask = loader_output['d_polarity_mask']
        d_event_list = loader_output['d_event_list']
        dt = loader_output['sampled_dt']


        print(f"File index: {file_index}")
        print(f"Sequence ID: {sequence_id}")
        print(f"Intrinsics: {intrinsics}")
        print(f"Event representation shape: {event_representation.shape}")
        print(f"Polarity mask shape: {polarity_mask.shape}")
        print(f"event_list: {raw_events.shape}")
        print(f"d_polarity_mask: {d_pol_mask.shape}")
        print(f"d_event_list: {d_event_list.shape}")
        print(f"Sampled dt: {dt}")
        print("------------------------------------------------------")


