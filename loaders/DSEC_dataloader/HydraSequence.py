from .FlowSequence import FlowSequence
from pathlib import Path
import numpy as np
import time
import torch
from ..utils.utils_augmentations import EventListAugmentor
from ..utils.img_event_ref_transform_DSEC import TransformImageToEventRef


class HydraSequence(FlowSequence):
    def __init__(self, augment=[], augment_prob=[], multiple_batches=True, **kwargs):
        super().__init__(**kwargs)

        self.augment = augment
        self.augument_prob = augment_prob
        if self.max_num_grad_events is None:
            self.max_num_detach_events = None
        else:
            self.max_num_detach_events = self.max_num_grad_events * 10

        augmentations = {}
        for aug, aug_prob in zip(self.augment, self.augument_prob):
            if aug not in ["Horizontal", "Vertical", "Polarity"]:
                raise ValueError(f"Unknown augmentation mechanism: {aug}")
            if aug == "Horizontal":
                augmentations["hflip_prob"] = aug_prob
            elif aug == "Vertical":
                augmentations["vflip_prob"] = aug_prob
            elif aug == "Polarity":
                augmentations["polarity_flip_prob"] = aug_prob
        self.event_augment = EventListAugmentor(width=self.width, height=self.height, **augmentations)
        # predict future time in milliseconds [ms] e.g. 10, 100, 1000 ms
        self.dt_ms = np.linspace(self.dt_ms[0], self.dt_ms[-1], num=1000, dtype=int) 

        self.img_transf = TransformImageToEventRef(conf=self.intrinsics, height=1080, width=1440)
        self.multiple_batches = multiple_batches


    def __len__(self):
        return len(self.timestamps) - 1

    def find_corresponding_flow_index(self, index):
        current_tstamp = self.timestamps[index]
        flow_index = np.where(self.timestamps_flow == current_tstamp)
        flow_index  = flow_index[0]
        if flow_index.size == 0:
            return None
        return flow_index.item()

    def _get_sequence_indices(self, index):
        # <= to avoid loading index 0
        if index <= self.sequence_len +1:
            # if the first index is negative use the forward sequence
            indices = list(range(index, index + self.sequence_len + 1))
            return indices
        start_ind = index - self.sequence_len - 1
        indices = list(range(start_ind, index))
        return indices

    def augment_events(self, xs, ys, ps, rec_xs, rec_ys, batch):
        """
        Augment event sequence with horizontal, vertical, and polarity flips.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ps: [N] tensor with event polarity ([-1, 1])
        :param rec_xs: [N] tensor with rectified event x location
        :param rec_ys: [N] tensor with rectified event y location
        :param batch: batch index
        :return xs: [N] tensor with augmented event x location
        :return ys: [N] tensor with augmented event y location
        :return ps: [N] tensor with augmented event polarity ([-1, 1])
        :return rec_xs: [N] tensor with augmented rectified event x location
        :return rec_ys: [N] tensor with augmented rectified event y location
        """

        for _, mechanism in enumerate(self.augment):

            if mechanism == "Horizontal":
                if self.batch_augmentation["Horizontal"][batch]:
                    xs = self.width - 1 - xs
                    if rec_xs is not None:
                        rec_xs = self.width - 1 - rec_xs

            elif mechanism == "Vertical":
                if self.batch_augmentation["Vertical"][batch]:
                    ys = self.height - 1 - ys
                    if rec_ys is not None:
                        rec_ys = self.height - 1 - rec_ys

            elif mechanism == "Polarity":
                if self.batch_augmentation["Polarity"][batch]:
                    ps *= -1

        return xs, ys, ps, rec_xs, rec_ys
    
    def flip_flow_direction(self, flow):
        # assumes flow of shape 2 x H x W
        if self.event_augment.last_flip_state["hflip"]:
            flow[0] *= -1
        if self.event_augment.last_flip_state["vflip"]:
            flow[1] *= -1
        return flow
    
    @staticmethod
    def sample_with_equal_prob(size, num_samples, replacement=False):
        if replacement:
            sampled = torch.randint(low=0, high=size, size=(num_samples,), dtype=torch.long)
        else:
            sampled = torch.randperm(size)[:num_samples]
        return sampled
  
    def _split_event_list(self, event_list, event_list_pol_mask):
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

        # TODO load always the same number of events for batching purpose
        num_of_events = event_list.shape[1]

        # Early return when the batch size is 1, as we don't need to preserve the batch shape
        if not self.multiple_batches:
            rand_indices = torch.randperm(num_of_events)
            grad_indices = rand_indices[:self.max_num_grad_events]
            detached_indices = rand_indices[self.max_num_grad_events:]

            event_list_ = event_list[:, grad_indices]
            event_list_pol_mask_ = event_list_pol_mask[:, grad_indices]
            d_event_list = event_list[:, detached_indices]
            d_event_list_pol_mask = event_list_pol_mask[:, detached_indices]
            return event_list_, event_list_pol_mask_, d_event_list, d_event_list_pol_mask

        if self.max_num_grad_events is None:
            return event_list, event_list_pol_mask, torch.zeros((4, 0)), torch.zeros((2, 0))

        if num_of_events > self.max_num_grad_events:
            indices = self.sample_with_equal_prob(num_of_events, self.max_num_grad_events, replacement=False)
        else:
            print("Not enough events for GRADIENT, sampling with replacement")
            indices = self.sample_with_equal_prob(num_of_events, self.max_num_grad_events, replacement=True)

        event_list_ = event_list[:, indices]
        event_list_pol_mask_ = event_list_pol_mask[:, indices]
        
        unsampled_indices = torch.ones(num_of_events, dtype=torch.bool)
        unsampled_indices[indices] = False
        unsampled_event_list = event_list[:, unsampled_indices]
        unsampled_event_list_pol_mask = event_list_pol_mask[:, unsampled_indices]
        num_of_unsampled_events = unsampled_event_list.shape[1]

        if num_of_unsampled_events > self.max_num_detach_events:
            d_indices = self.sample_with_equal_prob(num_of_unsampled_events, self.max_num_detach_events, replacement=False)
        elif num_of_unsampled_events == 0:
            # sample event from the original list multiple times
            # if there are no remaining events
            d_indices = self.sample_with_equal_prob(num_of_events, self.max_num_detach_events, replacement=True)
            unsampled_event_list = event_list
            unsampled_event_list_pol_mask = event_list_pol_mask
        else:
            print("Not enough events for LOSS COMPUTE, sampling with replacement")
            d_indices = self.sample_with_equal_prob(num_of_unsampled_events, self.max_num_detach_events, replacement=True)
        
        d_event_list = unsampled_event_list[:, d_indices]
        d_event_list_pol_mask = unsampled_event_list_pol_mask[:, d_indices]
        
        return event_list_, event_list_pol_mask_, d_event_list, d_event_list_pol_mask

    def split_events(self, x_rect, y_rect, p, t):
        p = self.norm_p(p)
        t = self.norm_t(t)

        # Split the event list into two lists, one of them 
        # (with max. length) to be used for backprop the other just for loss computation
        ev_list_pol_mask = self.create_polarity_mask(p)
        raw_ev = torch.from_numpy(np.stack([t, y_rect, x_rect, p], axis=0))

        # Split events for backprop and events just for calculation (detached)
        # TODO load always the same number of events for batching purpose
        ev_list, pol_mask, d_ev_list, d_pol_mask = self._split_event_list(raw_ev, ev_list_pol_mask)

        # Flip to have the same convention of the Iterative Warping loss
        ev_list = ev_list.permute(1, 0)
        pol_mask = pol_mask.permute(1, 0)
        d_ev_list = d_ev_list.permute(1, 0)
        d_pol_mask = d_pol_mask.permute(1, 0)
        return ev_list, pol_mask, d_ev_list, d_pol_mask
        
    def frame_gt(self, index):
        frame = super().frame_gt(index)
        # frame = frame.permute
        transformed_frame = self.img_transf(frame.permute(1,2,0).numpy())
        transformed_frame = transformed_frame[:480, :640, :]
        # Convert to tensor and change channel order
        return torch.from_numpy(transformed_frame).permute(2, 0, 1)
    
    def _get_single_datapoint_by_events(self, idx, events):
        disp_gt_path = Path(self.disp_gt_pathstrings[idx])
        file_index = int(disp_gt_path.stem)

        dyn_mask_gt_t0_path = Path(self.dyn_masks_pathstrings[idx])
        dynamic_mask_t0 = self.get_dynamic_obj_mask(dyn_mask_gt_t0_path)
        dynamic_mask_t0 = torch.from_numpy(dynamic_mask_t0).float().unsqueeze(0)
            
        x_0 = events['x_0']
        y_0 = events['y_0']
        p_0 = events['p_0']
        t_0 = events['t_0']

        # Augment events
        p_0 = p_0.astype(int)
        event_representation = self.get_event_representation(x_0, y_0, p_0, t_0)

        # --> Load future events for training future optical flow unsupervised
        sampled_dt = np.random.choice(self.dt_ms, size=1, replace=False)
        t_1 = t_0[0] + sampled_dt
        closest_ev_ind = np.argmin(np.abs(t_0 - t_1))

        x_1 = x_0[:closest_ev_ind]
        y_1 = y_0[:closest_ev_ind]
        p_1 = p_0[:closest_ev_ind]
        t_1 = t_0[:closest_ev_ind]

        event_list, pol_mask, d_event_list, d_pol_mask = self.split_events(
            x_rect=x_1, y_rect=y_1, p=p_1, t=t_1
            )
        event_list = event_list.float()
        pol_mask = pol_mask.float()
        d_event_list = d_event_list.float()
        d_pol_mask = d_pol_mask.float()
        
        frame = self.frame_gt(idx)

        data = {
            'file_index': file_index,
            'sequence_id': self.sequence_id,
            'representation': event_representation,
            'event_list': event_list, # N X 4
            'polarity_mask': pol_mask, # N X 2
            'd_event_list': d_event_list, # N X 4
            'd_polarity_mask': d_pol_mask, # N X 2
            'sampled_dt': sampled_dt/1000,
            'dynamic_mask': dynamic_mask_t0,
            'has_flow': torch.tensor(False),
            'forward_flow_gt': torch.zeros(2, self.height, self.width),
            'backward_flow_gt': torch.zeros(2, self.height, self.width),
            'frame': frame,
        }

        if self.flow_exists:
            flow_ind_t0 = self.find_corresponding_flow_index(idx)
            if flow_ind_t0 is not None:
                forward_flow = self.forward_flow_gt(flow_ind_t0)
                # backward_flow = self.backward_flow_gt(flow_ind_t0)

                forward_flow = torch.from_numpy(forward_flow)
                # backward_flow = torch.from_numpy(backward_flow)
                forward_flow = forward_flow.permute(2, 0, 1)
                # backward_flow = backward_flow.permute(2, 0, 1)

                data['has_flow'] = torch.tensor(True)
                data['forward_flow_gt'] = forward_flow
                # data['backward_flow_gt'] = backward_flow

        # Augment the sequence
        flow1 = data["forward_flow_gt"]
        flow2 = data["backward_flow_gt"]
        mask = data["dynamic_mask"]

        flow1, flow2, mask = self.event_augment.augment_dense_data(flow1=flow1, flow2=flow2, mask=mask)
        flow1 = self.flip_flow_direction(flow1)
        flow2 = self.flip_flow_direction(flow2)

        data["forward_flow_gt"] = flow1
        data["backward_flow_gt"] = flow2
        data["dynamic_mask"] = mask

        return data
    
    def preload_events(self, indices):
        t_s = self.timestamps[indices[0]] - self.delta_t_us
        t_e = self.timestamps[indices[-1]]
        x, y, p, t = self.get_events_start_end_time(t_s, t_e)
        
        # rectify events with bilinear interpolation
        x, y = self.get_rectified_events(x, y)

        # Augment events
        x, y, p = self.event_augment(x=x, y=y, p=p)

        t_ends = self.timestamps[indices]
        t_starts = t_ends - self.delta_t_us
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

    def __getitem__(self, index):
        index += 1 if index == 0 else 0
        self.indices = self._get_sequence_indices(index)
        events = self.preload_events(self.indices)

        sequence_data = []
        for i, idx in enumerate(self.indices):
            data = self._get_single_datapoint_by_events(idx=idx, events=events[i])
            sequence_data.append(data)

        if len(sequence_data) != self.sequence_len + 1:
            # return None  # skip inconsistent sample
            raise ValueError("Size mismatch in HydraSequence between sequence length and sequence data")
        
        return sequence_data



if __name__ == '__main__':
    # seq_abs_path = Path("//home/rpg/Downloads/DSEC/train/interlaken_00_c")
    # seq_abs_path = Path("/home/rpg/Downloads/DSEC/train/thun_00_a")
    seq_abs_path = Path("/data/scratch/pellerito/datasets/DSEC/train/thun_00_a")
    # dsec_seq = HydraSequence(
    #     seq_path=seq_abs_path, 
    #     num_bins=2, 
    #     sequence_len=2, 
    #     max_num_grad_events=10000, 
    #     dt=[1, 10, 100], 
    #     augment=["Horizontal", "Vertical", "Polarity"], 
    #     augment_prob=[0.5, 0.5, 0.5]
    #     )
    dsec_seq = HydraSequence(
        seq_path=seq_abs_path, 
        num_bins=2, 
        sequence_len=2, 
        max_num_grad_events=10000, 
        dt=[1, 10, 100]
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
            file_index = loader_output['file_index']
            sequence_id = loader_output['sequence_id']
            event_representation = loader_output['representation']
            dt = loader_output['sampled_dt']
            forward_flow_gt = loader_output['forward_flow_gt']
            backward_flow_gt = loader_output['backward_flow_gt']
            dynamic_mask = loader_output['dynamic_mask']

            event_list = loader_output['event_list']
            polarity_mask = loader_output['polarity_mask']
            d_event_list = loader_output['d_event_list']
            d_polarity_mask = loader_output['d_polarity_mask']
            
            frame = loader_output['frame']

            # print(f"event list shape: {event_list.shape}")
            # print(f"Polarity mask shape: {polarity_mask.shape}")
            # print(f"d_event_list: {d_event_list.shape}")
            # print(f"d_polarity_mask: {d_polarity_mask.shape}")
            # print(f"Forward flow GT is zero: {(forward_flow_gt == 0).all().item()}")
            # print(f"Backward flow GT is zero: {(backward_flow_gt == 0).all().item()}")
            print(f"-------------------------{time.time()-tm*1000}-----------------------------")


