from ..representations.representations import create_cnt_encoding
from .FlowSequence import FlowSequence
from pathlib import Path
import numpy as np
import torch


class FlowTestSequence(FlowSequence):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index):

        ts_end = self.timestamps[index]
        ts_start = ts_end - self.delta_t_us

        disp_gt_path = Path(self.disp_gt_pathstrings[index])
        file_index = int(disp_gt_path.stem)

        x,y,p,t = self.get_events_start_end_time(ts_start, ts_end)

        event_representation = self.get_cnt(x_=x, y_=y, p_=p)

        p = self.norm_p(p)
        t = self.norm_t(t)
        x_rect, y_rect = self.get_rectified_events(x, y)
        polarity_mask = self.create_polarity_mask(p)
        event_list = torch.from_numpy(np.stack([t, y_rect, x_rect, p], axis=0))

        # Flip to have the same convention of the Iterative Warping loss
        event_list = event_list.permute(1, 0)
        polarity_mask = polarity_mask.permute(1, 0)

        data = {
            'file_index': file_index,
            'sequence_id': self.sequence_id,
            'intrinsics': self.K,
            'representation': {"left": event_representation},
            'event_list': event_list, # N X 4
            'polarity_mask': polarity_mask, # N X 2
        }    

        if self.flow_exists:
            current_tstamp = self.timestamps[index]
            flow_index  = np.argmin(np.abs(self.timestamps_flow - current_tstamp))
            forward_flow = self.forward_flow_gt(flow_index)
            backward_flow = self.backward_flow_gt(flow_index)
            data['forward_flow_gt'] = forward_flow
            data['backward_flow_gt'] = backward_flow

        return data


if __name__ == '__main__':
    # seq_abs_path = Path("//home/rpg/Downloads/DSEC/train/interlaken_00_c")
    seq_abs_path = Path("/home/rpg/Downloads/DSEC/train/thun_00_a")
    dsec_seq = FlowSequence(seq_path=seq_abs_path, num_bins=2, sequence_len=2, max_num_grad_events=10000)
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


        print(f"File index: {file_index}")
        print(f"Sequence ID: {sequence_id}")
        print(f"Intrinsics: {intrinsics}")
        print(f"Event representation shape: {event_representation.shape}")
        print(f"Polarity mask shape: {polarity_mask.shape}")
        print(f"event_list: {raw_events.shape}")
        print(f"d_polarity_mask: {d_pol_mask.shape}")
        print(f"d_event_list: {d_event_list.shape}")
        print("------------------------------------------------------")


