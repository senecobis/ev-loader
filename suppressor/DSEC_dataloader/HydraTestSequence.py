from .HydraSequence import HydraSequence
from pathlib import Path
import numpy as np
import torch


class HydraTestSequence(HydraSequence):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, idx):
        disp_gt_path = Path(self.disp_gt_pathstrings[idx])
        file_index = int(disp_gt_path.stem)

        dyn_mask_gt_t0_path = Path(self.dyn_masks_pathstrings[idx])
        dynamic_mask = self.get_dynamic_obj_mask(dyn_mask_gt_t0_path)
        dynamic_mask = torch.from_numpy(dynamic_mask).float().unsqueeze(0)
        
        # |ts_start----ts_mid_point----ts_end|
        ts_end = self.timestamps[idx]
        ts_start = ts_end - self.delta_t_us
        x,y,p,t = self.get_events_start_end_time(ts_start, ts_end)
        x_rect, y_rect = self.get_rectified_events(x, y)
        event_representation = self.get_event_representation(x_rect, y_rect, p, t)

        frame = self.frame_gt(idx)

        data = {
            'file_index': file_index,
            'sequence_id': self.sequence_id,
            'representation': {"left": event_representation},
            'forward_flow_gt': None,
            'backward_flow_gt': None,
            'dynamic_mask': dynamic_mask,
            'frame': frame,
            'events': np.stack([x, y, t, p]).T
        }

        if self.flow_exists:
            flow_ind_t0 = self.find_corresponding_flow_index(idx)
            if flow_ind_t0 is None:
                return data
            forward_flow = self.forward_flow_gt(flow_ind_t0)
            backward_flow = self.backward_flow_gt(flow_ind_t0)

            forward_flow = torch.from_numpy(forward_flow)
            backward_flow = torch.from_numpy(backward_flow)
            forward_flow = forward_flow.permute(2, 0, 1)
            backward_flow = backward_flow.permute(2, 0, 1)

            data['forward_flow_gt'] = forward_flow
            data['backward_flow_gt'] = backward_flow

        return data


if __name__ == '__main__':
    # seq_abs_path = Path("/home/rpg/Downloads/DSEC/train/thun_00_a")
    seq_abs_path = Path("/data/scratch/pellerito/datasets/DSEC/train/thun_00_a")
    dsec_seq = HydraTestSequence(seq_path=seq_abs_path, num_bins=2)

    for loader_output in dsec_seq:
        file_index = loader_output['file_index']
        sequence_id = loader_output['sequence_id']
        event_representation = loader_output['representation']["left"]
        forward_flow_gt = loader_output['forward_flow_gt']
        backward_flow_gt = loader_output['backward_flow_gt']
        dynamic_mask = loader_output['dynamic_mask']
        events = loader_output['events']

        # print(f"File index: {file_index}")
        # print(f"Sequence ID: {sequence_id}")
        # print(f"Event representation shape: {event_representation.shape}")
        # print(f"Forward flow GT: {forward_flow_gt}")
        # print(f"Backward flow GT: {backward_flow_gt}")
        # print(f"Dynamic mask: {dynamic_mask.shape}")
        print(f"ev shape: {events.shape}")
        print("------------------------------------------------------")


