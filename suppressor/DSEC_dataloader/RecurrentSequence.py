from suppressor.DSEC_dataloader.sequence import Sequence
from pathlib import Path
import torch


class RecurrentSequence(Sequence):
    def __init__(self, sequence_len:int = 2, **kwargs):
        super().__init__(**kwargs)
        self.sequence_len = sequence_len

    def __getitem__(self, index):
        if index < self.sequence_len-1:
            # the actual index is smaller than the sequence length so we cannot create a sequence
            # we will return the first valid sequence, hence the index would be moved forward
            index = self.sequence_len-1
        indices = list(range(max(0, index - self.sequence_len + 1), index + 1))

        sequence_data = []
        for idx in indices:
            ts_end = self.timestamps[idx]
            ts_start = ts_end - self.delta_t_us

            disp_gt_path = Path(self.disp_gt_pathstrings[idx])
            dyn_mask_gt_path = Path(self.dyn_masks_pathstrings[idx])
            file_index = int(disp_gt_path.stem)

            disparity = self.get_disparity_map(disp_gt_path)
            depth = self.disparities_to_depths(disparity=disparity, Q=self.Q)
            dynamic_mask = self.get_dynamic_obj_mask(dyn_mask_gt_path)

            disparity = torch.from_numpy(disparity).float()
            dynamic_mask = torch.from_numpy(dynamic_mask).float().unsqueeze(0)
            depth = torch.from_numpy(depth).float()

            positive_pixels_percentage = ((dynamic_mask > 0).sum() / self.total_pixels_num)*100
            negative_pixels_percentage = ((dynamic_mask == 0).sum() / self.total_pixels_num)*100

            x_rect, y_rect, p, t = self.get_rectified_events_start_end_time(ts_start, ts_end)
            event_representation = self.get_event_representation(x_rect, y_rect, p, t)

            data = {
                'file_index': file_index,
                'sequence_id': self.sequence_id,
                'disparity_gt': disparity,
                'dynamic_mask_gt': dynamic_mask,
                'depth_gt': depth,
                'intrinsics': self.K,
                'representation': {"left": event_representation},
                'positive_pixels_percentage': positive_pixels_percentage,
                'negative_pixels_percentage': negative_pixels_percentage,
            }    
            sequence_data.append(data)

        return sequence_data


if __name__ == '__main__':
    # seq_abs_path = Path("//home/rpg/Downloads/DSEC/train/interlaken_00_c")
    seq_abs_path = Path("/home/rpg/Downloads/DSEC/train/thun_00_a")
    dsec_seq = RecurrentSequence(seq_path=seq_abs_path, num_bins=2, sequence_len=2)
    loader_outputs = dsec_seq[0]

    for loader_output in loader_outputs:
        file_index = loader_output['file_index']
        sequence_id = loader_output['sequence_id']
        disparity_gt = loader_output['disparity_gt']
        dynamic_mask_gt = loader_output['dynamic_mask_gt']
        depth_gt = loader_output['depth_gt']
        intrinsics = loader_output['intrinsics']
        # poses = loader_outputs['poses'] # poses at time t-1 and t
        event_representation = loader_output['representation']["left"]

        forward_flow_gt = loader_output['forward_flow_gt']
        backward_flow_gt = loader_output['backward_flow_gt']

        print(f"File index: {file_index}")
        print(f"Sequence ID: {sequence_id}")
        print(f"Disparity GT shape: {disparity_gt.shape}")
        print(f"Dynamic mask GT shape: {dynamic_mask_gt.shape}")
        print(f"Depth GT shape: {depth_gt.shape}")
        print(f"Intrinsics: {intrinsics}")
        # print(f"Poses at t-1: {poses[0]}")
        # print(f"Poses at t: {poses[1]}")
        print(f"Event representation shape: {event_representation.shape}")
        print(f"Positive pixels percentage: {loader_output['positive_pixels_percentage']}")
        print(f"Negative pixels percentage: {loader_output['negative_pixels_percentage']}")
        print(f"Forward flow GT: {forward_flow_gt.shape}")
        print(f"Backward flow GT: {backward_flow_gt.shape}")

