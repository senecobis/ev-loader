import time
import torch
from pathlib import Path
from collections import defaultdict
import numpy as np
from ..DSEC_dataloader.HydraSequence import HydraSequence  # Adjust as needed


def benchmark_hydra_sequence(seq_abs_path, num_samples=10):
    dsec_seq = HydraSequence(
        seq_path=seq_abs_path,
        num_bins=2,
        sequence_len=2,
        max_num_grad_events=10000,
        dt=[1, 10, 100],
        augment=["Horizontal", "Vertical", "Polarity"],
        augment_prob=[0.5, 0.5, 0.5],
        representation='stack'
    )

    timings = defaultdict(list)

    for ind in range(num_samples):
        index = ind + 1  # Avoid index 0 if needed
        timings_sample = {}

        t0 = time.time()
        indices = dsec_seq._get_sequence_indices(index)
        timings_sample["sequence_indices"] = time.time() - t0

        t1 = time.time()
        events = dsec_seq.preload_events(indices)
        timings_sample["preload_events"] = time.time() - t1

        sequence_data = []
        for i, idx in enumerate(indices):
            t2 = time.time()

            t_sub = time.time()
            disp_gt_path = Path(dsec_seq.disp_gt_pathstrings[idx])
            file_index = int(disp_gt_path.stem)
            t_flow_mask = time.time() - t_sub

            t_sub = time.time()
            dyn_mask_gt_t0_path = Path(dsec_seq.dyn_masks_pathstrings[idx])
            dynamic_mask_t0 = dsec_seq.get_dynamic_obj_mask(dyn_mask_gt_t0_path)
            dynamic_mask_t0 = torch.from_numpy(dynamic_mask_t0).float().unsqueeze(0)
            t_dynamic_mask = time.time() - t_sub

            t_sub = time.time()
            x_0 = events[i]['x_0']
            y_0 = events[i]['y_0']
            p_0 = events[i]['p_0']
            t_0 = events[i]['t_0']
            p_0 = p_0.astype(int)
            event_representation = dsec_seq.get_event_representation(x_0, y_0, p_0, t_0)
            t_event_rep = time.time() - t_sub

            t_sub = time.time()
            sampled_dt = np.random.choice(dsec_seq.dt_ms, size=1, replace=False)
            t1_time = t_0[0] + sampled_dt
            closest_ev_ind = np.argmin(np.abs(t_0 - t1_time))
            x_1 = x_0[:closest_ev_ind]
            y_1 = y_0[:closest_ev_ind]
            p_1 = p_0[:closest_ev_ind]
            t_1 = t_0[:closest_ev_ind]
            t_sampling = time.time() - t_sub

            t_sub = time.time()
            event_list, pol_mask, d_event_list, d_pol_mask = dsec_seq.split_events(
                x_rect=x_1, y_rect=y_1, p=p_1, t=t_1
            )
            t_event_split = time.time() - t_sub

            t_sub = time.time()
            frame = dsec_seq.frame_gt(idx)
            t_frame = time.time() - t_sub

            t_sub = time.time()
            has_flow = False
            forward_flow = torch.zeros(2, dsec_seq.height, dsec_seq.width)
            backward_flow = torch.zeros(2, dsec_seq.height, dsec_seq.width)

            if dsec_seq.flow_exists:
                flow_ind_t0 = dsec_seq.find_corresponding_flow_index(idx)
                if flow_ind_t0 is not None:
                    f_flow = dsec_seq.forward_flow_gt(flow_ind_t0)
                    b_flow = dsec_seq.backward_flow_gt(flow_ind_t0)
                    f_flow = torch.from_numpy(f_flow).permute(2, 0, 1)
                    b_flow = torch.from_numpy(b_flow).permute(2, 0, 1)
                    forward_flow = f_flow
                    backward_flow = b_flow
                    has_flow = True
            t_flow_loading = time.time() - t_sub

            t_sub = time.time()
            flow1, flow2, mask = dsec_seq.event_augment.augment_dense_data(
                flow1=forward_flow, flow2=backward_flow, mask=dynamic_mask_t0
            )
            flow1 = dsec_seq.flip_flow_direction(flow1)
            flow2 = dsec_seq.flip_flow_direction(flow2)
            t_dense_aug = time.time() - t_sub

            timings_sample[f"single_datapoint_{i}"] = time.time() - t2
            timings_sample[f"flow_mask_path_{i}"] = t_flow_mask
            timings_sample[f"dyn_mask_load_{i}"] = t_dynamic_mask
            timings_sample[f"event_repr_{i}"] = t_event_rep
            timings_sample[f"event_sampling_{i}"] = t_sampling
            timings_sample[f"event_split_{i}"] = t_event_split
            timings_sample[f"frame_load_{i}"] = t_frame
            timings_sample[f"flow_load_{i}"] = t_flow_loading
            timings_sample[f"dense_aug_{i}"] = t_dense_aug

            sequence_data.append("done")  # placeholder

        total_time = sum([v for k, v in timings_sample.items() if k.startswith("single_datapoint_")])
        timings_sample["total"] = timings_sample["preload_events"] + total_time

        print(f"\nSample {ind}:")
        for key, val in timings_sample.items():
            print(f"  {key}: {val:.4f} sec")
        print("------------------------------------------------------")

        for key, val in timings_sample.items():
            timings[key].append(val)

    print("\nAverage timings over", num_samples, "samples:")
    for key, val_list in timings.items():
        print(f"  {key}: {np.mean(val_list):.4f} sec")


if __name__ == "__main__":
    seq_abs_path = Path("/data/scratch/pellerito/datasets/DSEC/train/thun_00_a")
    benchmark_hydra_sequence(seq_abs_path, num_samples=10)
