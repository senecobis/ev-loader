import os
import sys
import cv2
import torch
import kornia
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from mot_seg_trainer import MotionSegmentationModule
from homo_mot_seg_trainer import HomoAndMotSegModule
from homo_flow_mot_seg_trainer import HomoFlowAndMotSegModule
from homo_flow_temp_mot_seg_trainer import HomoFlowAndTempMotSegModule
from robosense.dataset.representations import StackOfEventFrames


if __name__ == "__main__":
    num_bins = 100
    height = 260
    width = 346

    evimo_path = "/home/stamatios/SSD2/EVIMO"
    metadata = np.load(os.path.join(evimo_path, "metadata_txt.npz"))
    metadata = metadata["eval"]

    # --- MOTION SEGMENTATION ---

    # mot_seg_module = MotionSegmentationModule.load_from_checkpoint(
    #    "/home/stamatios/HDD2/3dv_exps/evimo/mot_seg/evs_resnet34d/version_1/checkpoints/epoch=39-step=313520.ckpt",
    #    in_channels=10,
    # )
    # mot_seg_module.eval().to(device="cpu")

    # --- HOMOGRAPHY + MOTION SEGMENTATION ---

    homo_mot_seg_module = HomoAndMotSegModule.load_from_checkpoint(
        "/home/stamatios/HDD2/3dv_exps/evimo/homo_mot_seg/evs_resnet34d/version_0/checkpoints/epoch=37-step=297844.ckpt",
        homo_checkpoint="/home/stamatios/HDD2/3dv_exps/evimo/homo/evs_resnet34d/version_0/checkpoints/epoch=26-step=211626.ckpt",
    )
    homo_mot_seg_module.eval().to(device="cpu")

    # --- HOMOGRAPHY + FLOW + MOTION SEGMENTATION ---

    # homo_flow_mot_seg_module = HomoFlowAndMotSegModule.load_from_checkpoint(
    #    '/home/stamatios/HDD2/3dv_exps/evimo/homo_flow_mot_seg/evs_resnet34d/version_0/checkpoints/epoch=39-step=313520.ckpt',
    #    homo_checkpoint='/home/stamatios/HDD2/3dv_exps/evimo/homo/evs_resnet34d/version_0/checkpoints/epoch=26-step=211626.ckpt',
    #    flow_checkpoint='/home/stamatios/HDD2/3dv_exps/evimo/flow/evs_resnet34d/version_0/checkpoints/epoch=27-step=219464.ckpt',
    # )
    # homo_flow_mot_seg_module.eval().to(device="cpu")

    # --- HOMOGRAPHY + FLOW + TEMPORAL MOTION SEGMENTATION ---

    # homo_flow_temp_mot_seg_module = HomoFlowAndTempMotSegModule.load_from_checkpoint(
    #     "/home/stamatios/HDD2/3dv_exps/evimo/homo_flow_temp_mot_seg/evs_resnet34d/version_0/checkpoints/epoch=35-step=282168.ckpt",
    #     homo_checkpoint="/home/stamatios/HDD2/3dv_exps/evimo/homo/evs_resnet34d/version_0/checkpoints/epoch=26-step=211626.ckpt",
    #     flow_checkpoint="/home/stamatios/HDD2/3dv_exps/evimo/flow/evs_resnet34d/version_0/checkpoints/epoch=27-step=219464.ckpt",
    # )
    # homo_flow_temp_mot_seg_module.eval().to(device="cpu")

    event_frame_stack = StackOfEventFrames(channels=10, height=height, width=width)

    min_events = 30
    max_background_ratio = 2

    ious = []
    for index in tqdm(range(len(metadata)), position=0, leave=True):
        events_file = os.path.join(evimo_path, metadata[index])
        dir_name = os.path.dirname(events_file)
        file_id = int(events_file.split("_")[-1].split(".")[0])

        # events_file_prev = os.path.join(evimo_path, metadata[index - 1])
        # if not os.path.exists(events_file_prev):
        #    ious.append(-1)
        #    continue

        events = torch.load(events_file)

        ts = events[:, 0].numpy()
        ts = ((num_bins - 1) * (ts - ts[0]) / (ts[-1] - ts[0])).astype(np.int64)
        xs = events[:, 1].numpy().astype(np.int64)
        ys = events[:, 2].numpy().astype(np.int64)
        ps = events[:, 3].numpy().clip(0, 1).astype(np.int64)

        spike_tensor = torch.zeros((2, height, width, num_bins))
        spike_tensor[ps, ys, xs, ts] = 1

        full_mask_tensor = torch.zeros((2, height, width, num_bins))

        dmask_file_start = os.path.join(dir_name, f"depth_mask_{file_id}.png")
        dmask_file_end = dmask_file_start.replace(f"{file_id}", f"{file_id+1}")

        depth_mask_start = cv2.imread(dmask_file_start, -1)
        depth_mask_end = cv2.imread(dmask_file_end, -1)

        mask_start = depth_mask_start[:, :, 2].astype("float32") / 1000
        mask_start[mask_start > 0] = 1

        if mask_start.sum() < min_events:
            ious.append(-1)
            continue

        mask_end = depth_mask_end[:, :, 2].astype("float32") / 1000
        mask_end[mask_end > 0] = 1

        if mask_end.sum() < min_events:
            ious.append(-1)
            continue

        fullmask = np.stack([mask_start, mask_end], axis=0).max(axis=0)

        kernel = np.ones((5, 5), "uint8")
        fullmask = cv2.dilate(fullmask, kernel, iterations=1)

        full_mask_tensor = torch.from_numpy(
            np.tile(np.expand_dims(fullmask, axis=(0, 3)), (2, 1, 1, num_bins))
        )

        masked_spike_tensor = ((spike_tensor + full_mask_tensor) > 1).float()
        background_spikes = (
            spike_tensor + torch.logical_not(masked_spike_tensor).float()
        ) > 1

        ratio = torch.sum(background_spikes) / torch.sum(masked_spike_tensor)
        if ratio > max_background_ratio:
            ious.append(-1)
            continue

        with torch.no_grad():
            event_repr = (
                event_frame_stack.convert(
                    x=events[:, 1],
                    y=events[:, 2],
                    pol=events[:, 3],
                    time=events[:, 0],
                ).unsqueeze(0)
                - 0.5
            )

            B, C, H, W = event_repr.shape

            # --- MOTION SEGMENTATION ---

            # event_repr_bidi = torch.stack(
            #    [-torch.flip(event_repr, dims=[1]), event_repr], dim=1
            # ).view(B * 2, C, H, W)

            # spike_pred_2D = torch.sigmoid(
            #    mot_seg_module.mot_seg_net(event_repr_bidi).view(B, 2, H, W).squeeze(0)
            # ).max(dim=0)[0]

            # --- HOMOGRAPHY + MOTION SEGMENTATION ---

            homo_output = homo_mot_seg_module.homo_net(event_repr).view(-1, 4, 2)
            points_off = homo_mot_seg_module.homo_net.homo_alpha * torch.tanh(
                homo_output
            )
            points_dst = torch.tensor(
                [
                    [
                        [0.0, 0.0],
                        [W - 1.0, 0.0],
                        [W - 1.0, H - 1.0],
                        [0.0, H - 1.0],
                    ]
                ],
                dtype=event_repr.dtype,
                device=event_repr.device,
            ).repeat(B, 1, 1)
            points_src = points_dst + points_off * torch.tensor(
                [W - 1.0, H - 1.0],
                dtype=event_repr.dtype,
                device=event_repr.device,
            ).view(1, 1, 2)
            homographies = kornia.geometry.get_perspective_transform(
                points_src, points_dst
            ).view(B, 1, 3 * 3)
            interp_weights = (
                torch.linspace(
                    0.0,
                    1.0,
                    C,
                    dtype=event_repr.dtype,
                    device=event_repr.device,
                )
                .repeat(B, 1)
                .unsqueeze(2)
            )
            M_eye = (
                torch.eye(
                    3,
                    dtype=event_repr.dtype,
                    device=event_repr.device,
                )
                .view(1, 1, 3 * 3)
                .repeat(B, 1, 1)
            )
            M_bwd = torch.bmm(1.0 - interp_weights, M_eye).view(
                B * C, 3, 3
            ) + torch.bmm(interp_weights, homographies).view(B * C, 3, 3)
            M_fwd = torch.flip(M_bwd.view(B, C, 3, 3), [1]).view(B * C, 3, 3).inverse()
            event_repr_bwd = kornia.geometry.warp_perspective(
                event_repr.view(B * C, 1, H, W),
                M_bwd,
                dsize=(H, W),
                padding_mode="zeros",
                align_corners=False,
            ).view(B, C, H, W)
            event_repr_fwd = kornia.geometry.warp_perspective(
                event_repr.view(B * C, 1, H, W),
                M_fwd,
                dsize=(H, W),
                padding_mode="zeros",
                align_corners=False,
            ).view(B, C, H, W)

            event_repr_bidi = torch.stack(
                [-torch.flip(event_repr_bwd, dims=[1]), event_repr_fwd], dim=1
            ).view(B * 2, C, H, W)

            spike_pred_2D = torch.sigmoid(
                homo_mot_seg_module.mot_seg_net(event_repr_bidi)
                .view(B, 2, H, W)
                .squeeze(0)
            ).max(dim=0)[0]

            # --- HOMOGRAPHY + FLOW + MOTION SEGMENTATION ---

            # homo_output = homo_flow_mot_seg_module.homo_net(event_repr).view(-1, 4, 2)
            # points_off = homo_flow_mot_seg_module.homo_net.homo_alpha * torch.tanh(
            #    homo_output
            # )
            # points_dst = torch.tensor(
            #    [
            #        [
            #            [0.0, 0.0],
            #            [W - 1.0, 0.0],
            #            [W - 1.0, H - 1.0],
            #            [0.0, H - 1.0],
            #        ]
            #    ],
            #    dtype=event_repr.dtype,
            #    device=event_repr.device,
            # ).repeat(B, 1, 1)
            # points_src = points_dst + points_off * torch.tensor(
            #    [W - 1.0, H - 1.0],
            #    dtype=event_repr.dtype,
            #    device=event_repr.device,
            # ).view(1, 1, 2)
            # homographies = kornia.geometry.get_perspective_transform(
            #    points_src, points_dst
            # ).view(B, 1, 3 * 3)
            # interp_weights = (
            #    torch.linspace(
            #        0.0,
            #        1.0,
            #        C,
            #        dtype=event_repr.dtype,
            #        device=event_repr.device,
            #    )
            #    .repeat(B, 1)
            #    .unsqueeze(2)
            # )
            # M_eye = (
            #    torch.eye(
            #        3,
            #        dtype=event_repr.dtype,
            #        device=event_repr.device,
            #    )
            #    .view(1, 1, 3 * 3)
            #    .repeat(B, 1, 1)
            # )
            # M_bwd = torch.bmm(1.0 - interp_weights, M_eye).view(
            #    B * C, 3, 3
            # ) + torch.bmm(interp_weights, homographies).view(B * C, 3, 3)
            # M_fwd = torch.flip(M_bwd.view(B, C, 3, 3), [1]).view(B * C, 3, 3).inverse()
            # event_repr_bwd = kornia.geometry.warp_perspective(
            #    event_repr.view(B * C, 1, H, W),
            #    M_bwd,
            #    dsize=(H, W),
            #    padding_mode="zeros",
            #    align_corners=False,
            # ).view(B, C, H, W)
            # event_repr_fwd = kornia.geometry.warp_perspective(
            #    event_repr.view(B * C, 1, H, W),
            #    M_fwd,
            #    dsize=(H, W),
            #    padding_mode="zeros",
            #    align_corners=False,
            # ).view(B, C, H, W)

            # event_repr_bidi = torch.stack(
            #    [-torch.flip(event_repr, dims=[1]), event_repr], dim=1
            # ).view(B * 2, C, H, W)

            # flow_output = homo_flow_mot_seg_module.flow_net(event_repr_bidi)
            # flow_pred_bwd = flow_output[:1]
            # flow_pred_fwd = flow_output[1:]

            # event_repr_bidi = torch.stack(
            #    [
            #        torch.cat(
            #            [-torch.flip(event_repr_bwd, dims=[1]), flow_pred_bwd], dim=1
            #        ),
            #        torch.cat([event_repr_fwd, flow_pred_fwd], dim=1),
            #    ],
            #    dim=1,
            # ).view(B * 2, C + 2, H, W)

            # spike_pred_2D = torch.sigmoid(
            #    homo_flow_mot_seg_module.mot_seg_net(event_repr_bidi)
            #    .view(B, 2, H, W)
            #    .squeeze(0)
            # ).max(dim=0)[0]

            # --- HOMOGRAPHY + FLOW + TEMPORAL MOTION SEGMENTATION ---

            # homo_output = homo_flow_temp_mot_seg_module.homo_net(event_repr).view(
            #     -1, 4, 2
            # )
            # points_off = homo_flow_temp_mot_seg_module.homo_net.homo_alpha * torch.tanh(
            #     homo_output
            # )
            # points_dst = torch.tensor(
            #     [
            #         [
            #             [0.0, 0.0],
            #             [W - 1.0, 0.0],
            #             [W - 1.0, H - 1.0],
            #             [0.0, H - 1.0],
            #         ]
            #     ],
            #     dtype=event_repr.dtype,
            #     device=event_repr.device,
            # ).repeat(B, 1, 1)
            # points_src = points_dst + points_off * torch.tensor(
            #     [W - 1.0, H - 1.0],
            #     dtype=event_repr.dtype,
            #     device=event_repr.device,
            # ).view(1, 1, 2)
            # homographies = kornia.geometry.get_perspective_transform(
            #     points_src, points_dst
            # ).view(B, 1, 3 * 3)
            # interp_weights = (
            #     torch.linspace(
            #         0.0,
            #         1.0,
            #         C,
            #         dtype=event_repr.dtype,
            #         device=event_repr.device,
            #     )
            #     .repeat(B, 1)
            #     .unsqueeze(2)
            # )
            # M_eye = (
            #     torch.eye(
            #         3,
            #         dtype=event_repr.dtype,
            #         device=event_repr.device,
            #     )
            #     .view(1, 1, 3 * 3)
            #     .repeat(B, 1, 1)
            # )
            # M_bwd = torch.bmm(1.0 - interp_weights, M_eye).view(
            #     B * C, 3, 3
            # ) + torch.bmm(interp_weights, homographies).view(B * C, 3, 3)
            # M_fwd = torch.flip(M_bwd.view(B, C, 3, 3), [1]).view(B * C, 3, 3).inverse()
            # event_repr_bwd = kornia.geometry.warp_perspective(
            #     event_repr.view(B * C, 1, H, W),
            #     M_bwd,
            #     dsize=(H, W),
            #     padding_mode="zeros",
            #     align_corners=False,
            # ).view(B, C, H, W)
            # event_repr_fwd = kornia.geometry.warp_perspective(
            #     event_repr.view(B * C, 1, H, W),
            #     M_fwd,
            #     dsize=(H, W),
            #     padding_mode="zeros",
            #     align_corners=False,
            # ).view(B, C, H, W)

            # event_repr_bidi = torch.stack(
            #     [-torch.flip(event_repr, dims=[1]), event_repr], dim=1
            # ).view(B * 2, C, H, W)

            # flow_output = homo_flow_temp_mot_seg_module.flow_net(event_repr_bidi)
            # flow_pred_bwd = flow_output[:1]
            # flow_pred_fwd = flow_output[1:]

            # events_prev = torch.load(events_file_prev)

            # event_repr_prev = (
            #     event_frame_stack.convert(
            #         x=events_prev[:, 1],
            #         y=events_prev[:, 2],
            #         pol=events_prev[:, 3],
            #         time=events_prev[:, 0],
            #     ).unsqueeze(0)
            #     - 0.5
            # )

            # homo_output = homo_flow_temp_mot_seg_module.homo_net(event_repr_prev).view(
            #     -1, 4, 2
            # )
            # points_off = homo_flow_temp_mot_seg_module.homo_net.homo_alpha * torch.tanh(
            #     homo_output
            # )
            # points_dst = torch.tensor(
            #     [
            #         [
            #             [0.0, 0.0],
            #             [W - 1.0, 0.0],
            #             [W - 1.0, H - 1.0],
            #             [0.0, H - 1.0],
            #         ]
            #     ],
            #     dtype=event_repr.dtype,
            #     device=event_repr.device,
            # ).repeat(B, 1, 1)
            # points_src = points_dst + points_off * torch.tensor(
            #     [W - 1.0, H - 1.0],
            #     dtype=event_repr.dtype,
            #     device=event_repr.device,
            # ).view(1, 1, 2)
            # homographies = kornia.geometry.get_perspective_transform(
            #     points_src, points_dst
            # ).view(B, 1, 3 * 3)
            # interp_weights = (
            #     torch.linspace(
            #         0.0,
            #         1.0,
            #         C,
            #         dtype=event_repr.dtype,
            #         device=event_repr.device,
            #     )
            #     .repeat(B, 1)
            #     .unsqueeze(2)
            # )
            # M_eye = (
            #     torch.eye(
            #         3,
            #         dtype=event_repr.dtype,
            #         device=event_repr.device,
            #     )
            #     .view(1, 1, 3 * 3)
            #     .repeat(B, 1, 1)
            # )
            # M_bwd = torch.bmm(1.0 - interp_weights, M_eye).view(
            #     B * C, 3, 3
            # ) + torch.bmm(interp_weights, homographies).view(B * C, 3, 3)
            # M_fwd = torch.flip(M_bwd.view(B, C, 3, 3), [1]).view(B * C, 3, 3).inverse()
            # event_repr_prev_bwd = kornia.geometry.warp_perspective(
            #     event_repr_prev.view(B * C, 1, H, W),
            #     M_bwd,
            #     dsize=(H, W),
            #     padding_mode="zeros",
            #     align_corners=False,
            # ).view(B, C, H, W)
            # event_repr_prev_fwd = kornia.geometry.warp_perspective(
            #     event_repr_prev.view(B * C, 1, H, W),
            #     M_fwd,
            #     dsize=(H, W),
            #     padding_mode="zeros",
            #     align_corners=False,
            # ).view(B, C, H, W)

            # event_repr_prev_bidi = torch.stack(
            #     [-torch.flip(event_repr_prev, dims=[1]), event_repr_prev], dim=1
            # ).view(B * 2, C, H, W)

            # flow_output = homo_flow_temp_mot_seg_module.flow_net(event_repr_prev_bidi)
            # flow_pred_prev_bwd = flow_output[:1]
            # flow_pred_prev_fwd = flow_output[1:]

            # event_repr_prev_bidi = torch.stack(
            #     [
            #         torch.cat(
            #             [
            #                 -torch.flip(event_repr_prev_bwd, dims=[1]),
            #                 flow_pred_prev_bwd,
            #             ],
            #             dim=1,
            #         ),
            #         torch.cat([event_repr_prev_fwd, flow_pred_prev_fwd], dim=1),
            #     ],
            #     dim=1,
            # ).view(B * 2, C + 2, H, W)

            # hidden_state = homo_flow_temp_mot_seg_module.mot_seg_net.encoder(
            #     event_repr_prev_bidi
            # )[-1]

            # event_repr_bidi = torch.stack(
            #     [
            #         torch.cat(
            #             [-torch.flip(event_repr_bwd, dims=[1]), flow_pred_bwd], dim=1
            #         ),
            #         torch.cat([event_repr_fwd, flow_pred_fwd], dim=1),
            #     ],
            #     dim=1,
            # ).view(B * 2, C + 2, H, W)

            # features = homo_flow_temp_mot_seg_module.mot_seg_net.encoder(
            #     event_repr_bidi
            # )
            # bottleneck = features[-1]
            # features[-1] = homo_flow_temp_mot_seg_module.CCT(
            #     bottleneck, hidden_state
            # ) + homo_flow_temp_mot_seg_module.CST(bottleneck, hidden_state)
            # out = homo_flow_temp_mot_seg_module.mot_seg_net.decoder(*features)

            # spike_pred_2D = torch.sigmoid(
            #     homo_flow_temp_mot_seg_module.mot_seg_net.head(out)
            #     .view(B, 2, H, W)
            #     .squeeze(0)
            # ).max(dim=0)[0]

            # pred_label = torch.sigmoid(
            #    mot_seg_module.mot_seg_net(event_repr_bidi).view(B, 2, H, W)
            # ).max(dim=1, keepdim=True)[0]

        spike_mask_2D = torch.sum(masked_spike_tensor, dim=(0, 3))

        spike_pred_2D = (
            spike_pred_2D.unsqueeze(0).unsqueeze(-1).repeat(2, 1, 1, num_bins)
        )
        spike_pred_2D[spike_pred_2D >= 0.5] = 1.0
        spike_pred_2D[spike_pred_2D < 0.5] = 0.0
        spike_pred_2D = ((spike_tensor + spike_pred_2D) > 1).float()
        spike_pred_2D = torch.sum(spike_pred_2D, dim=(0, 3))

        spike_pred = spike_pred_2D.numpy()
        spike_gt = spike_mask_2D.numpy()

        intersection = np.sum(np.logical_and(spike_pred, spike_gt))
        union = np.sum(np.logical_or(spike_pred, spike_gt))
        point_iou = intersection / union

        # ones = torch.ones_like(pred_label)
        # zeros = torch.zeros_like(pred_label)

        # pred_label = torch.where(pred_label >= 0.5, ones, zeros)

        # target_label = torch.from_numpy(fullmask).unsqueeze(0).unsqueeze(0)
        # target_label = torch.where(target_label > 0.5, ones, zeros)

        # binary_events = torch.where(spike_tensor.max(dim=0)[0].permute(2, 0, 1).unsqueeze(0) > 0, ones, zeros)
        # pred_events = binary_events * pred_label.repeat(1, num_bins, 1 ,1)
        # target_events = binary_events * target_label.repeat(1, num_bins, 1 ,1)

        # intersection = torch.sum(pred_events * target_events, dim=(1, 2, 3))
        # union = torch.sum(pred_events + target_events, dim=(1, 2, 3)) - intersection
        # point_iou = torch.mean(intersection / (union + 1e-6))

        ious.append(point_iou)

    iou = np.array(ious)
    np.save("iou.npy", iou)
    iou = iou[iou >= 0].mean()
    print("IoU: {}".format(iou))