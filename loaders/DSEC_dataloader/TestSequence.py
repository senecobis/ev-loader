from ..DSEC_dataloader.sequence import Sequence
from pathlib import Path
import numpy as np 
import torch

# TODO import events as they arrive (same of RAMPVO) and use the closest GT point as the target

class TestSequence(Sequence):
    def __init__(self, FPS=None, **kwargs):
        super().__init__(**kwargs)
        if FPS is not None:
            assert FPS >= self.FPS_GT, "FPS should be greater than FPS_GT"
            assert FPS % self.FPS_GT == 0, "FPS should be a multiple of FPS_GT"
            self.FPS = FPS
            self.up_timestamps = self.upscale_timestamps()
            self.data_sequence_len = len(self.up_timestamps)
            self.existing_GT_mask = torch.isin(torch.from_numpy(self.up_timestamps), torch.from_numpy(self.timestamps)) 
        else:
            self.FPS = self.FPS_GT
            self.data_sequence_len = len(self.timestamps)

    def upscale_timestamps(self):
        """Upscale the timestamps to the FPS
            The number of new samples should be 
            num_new_samples = (len(self.timestamps)-1)*int(scale_factor+1) + 1
        """
        scale_factor = round(self.FPS/self.FPS_GT)
        timestamps = self.timestamps
        new_timestamps = np.linspace(0, len(timestamps), (len(timestamps))*scale_factor + 1)
        indices = np.arange(0, len(timestamps))
        upscaled_timestamps = np.interp(new_timestamps, indices, timestamps)
        return upscaled_timestamps

    def __len__(self):
        return self.data_sequence_len
    
    def __getitem__(self, index):
        if self.FPS == self.FPS_GT:
            return super().__getitem__(index)
        else:
            return self.getitem_by_time(index)
    
    def getitem_by_time(self, index):
        """Specify an index of the sequence, based on the index if it corresponds to a GT point
            We return this point, otherwise we don't provide GT.
            This is useful when we want to inference at a specified rate (constant) which is > then GT rate
        """
        ts_end = self.up_timestamps[index]
        ts_start = ts_end - self.delta_t_us

        file_index = None
        disparity = None
        dynamic_mask = None
        depth = None
        
        is_GT_available = self.existing_GT_mask[index]
        if is_GT_available:
            file_index = self.file_index(index)
            dynamic_mask = self.dynamic_mask_gt(index)
            disparity = self.disparity_gt(index)
            depth = self.depth_gt(disparity)
            frame = self.frame_gt(index)

        x_rect, y_rect, p, t = self.get_rectified_events_start_end_time(ts_start, ts_end)
        event_representation = self.get_event_representation(x_rect, y_rect, p, t)

        output = {
            'file_index': file_index,
            'sequence_id': self.sequence_id,
            'dynamic_mask_gt': dynamic_mask,
            'depth_gt': depth,
            'intrinsics': self.K,
            'representation': {"left": event_representation},
            'frame': frame,
            'is_GT_available': is_GT_available
        }
        return output
    
if __name__ == '__main__':
    seq_abs_path = Path("/home/rpg/Downloads/DSEC/test/interlaken_00_a")
    dsec_seq = TestSequence(seq_path=seq_abs_path, num_bins=2, representation="raw", FPS=30)

    for ind, loader_output in enumerate(dsec_seq):
        file_index = loader_output['file_index']
        sequence_id = loader_output['sequence_id']
        dynamic_mask_gt = loader_output['dynamic_mask_gt']
        depth_gt = loader_output['depth_gt']
        intrinsics = loader_output['intrinsics']
        event_representation = loader_output['representation']["left"]

        if dynamic_mask_gt is not None:
            # print(f"File index: {file_index}")
            # print(f"Sequence ID: {sequence_id}")
            # print(f"Disparity GT shape: {disparity_gt}")
            # print(f"Dynamic mask GT shape: {dynamic_mask_gt}")
            # print(f"Depth GT shape: {depth_gt}")
            # print(f"Intrinsics: {intrinsics}")
            print(f"Event representation shape: {event_representation.shape}")
        else:
            print(f"No GT available {ind}")
