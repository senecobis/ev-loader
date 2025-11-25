""" This file contains the function to warp the events according to poses and depths."""

import copy
import numpy as np


def warp_events_poses_depths(events, depths, K, T_rel):
    """Warp events from one frame to another using depth information.
    Args:
        events (np.ndarray) ... N x 4 events in the reference frame.
        depths (np.ndarray) ... N x 2 the Depth image.
        K (np.ndarray) ... 3 x 3 Camera intrinsics.
        T_rel (np.ndarray) ... 4 x 4 Relative transformation between the two frames.
    
    Returns:
        warped_2d (np.ndarray) ... N x 4 Warped 2D events.
    """
    # Setp 0: Select events coordinates and corresponding depths
    events_ = copy.deepcopy(events)
    events_2d = events_[..., :2]
    N = len(events_2d)
    events_2d_int = np.round(events_2d).astype(int)
    depths_ = depths[(events_2d_int[:, 1], events_2d_int[:, 0])]

    # Step 1: Inverse projection (2D to 3D)
    events_2d_homogeneous = np.hstack((events_2d, np.ones((N, 1))))
    events_3d = depths_ * (np.linalg.inv(K) @ events_2d_homogeneous.T)

    # Step 2: Transform 3D points
    events_3d_homogeneous = np.hstack((events_3d.T, np.ones((N, 1))))
    transformed_3d = T_rel @ events_3d_homogeneous.T

    # Step 3: Project back to 2D
    warped_2d_homogeneous = (K @ transformed_3d[:3, :]).T
    warped_2d = warped_2d_homogeneous[:, :2] / warped_2d_homogeneous[:, 2:]

    events_[:,0] = warped_2d[:,0]
    events_[:,1] = warped_2d[:,1]

    return events_


def warp_according_to_optical_flow(batch, best_motion=None, dense_flow=None, mask=True, solv=None, image_shape=None):
    t_scale = batch[:, 2].max() - batch[:, 2].min()
    
    if best_motion is not None and dense_flow is None:
        dense_flow = solv.motion_to_dense_flow(best_motion, t_scale) * t_scale
    elif dense_flow is not None and best_motion is None:
        dense_flow *= t_scale
    else:
        raise ValueError("Either best_motion or dense_flow should be provided.")

    backward_events, backward_feat = solv.warper.warp_event(
        events=batch, motion=dense_flow, motion_model=solv.motion_model_for_dense_warp, direction="first"
    )
    if mask:
        mask = (backward_events[..., 0] > 0) & \
            (backward_events[..., 0] < image_shape[0]) & \
            (backward_events[..., 1] > 0) & \
            (backward_events[..., 1] < image_shape[1])

        backward_events = backward_events[mask]
    return backward_events
