import numpy as np
from suppressor.utils.utils_warp_events import warp_events_poses_depths
from suppressor.utils.utils_matrix_op import coordinates_flow3D


class RigidFlow():
    def __init__(self, events, depth, K, image_shape):
        """Calculate the rigid flow of the event stream given the depth at one instant and the 
           poses of the camera at t_first_event and t_last_event.

        Args:
            events (np.ndarray): N x 4 event stream to use.
            depths (np.ndarray): H x W depth image.
            K (np.ndarray): 3 x 3 Camera intrinsics.
            image_shape (tuple): Shape of the image (W, H).
        """
        self.events = events
        self.depth = depth
        self.K = K
        self.width, self.height = image_shape
        self.image_shape = image_shape
        if self.width < self.height:
            Warning("The image shape is (W, H) BE CAREFUL.")

    def warp_rigid(self, T_w0, T_w1):
        T_1_0 = np.linalg.inv(T_w1) @ T_w0
        warped_events = warp_events_poses_depths(events=self.events, 
                                        depths=self.depth, 
                                        K=self.K, 
                                        T_rel=T_1_0
                                        )
        warped_events_mask = ~np.isnan(warped_events).any(axis=1) & \
                    (warped_events[:, 0] >= 0) & \
                    (warped_events[:, 0] < self.width) & \
                    (warped_events[:, 1] >= 0) & \
                    (warped_events[:, 1] < self.height)
        warped_filtered = warped_events[warped_events_mask]

        self.warped_events_mask = warped_events_mask
        self.warped_events = warped_events
        return warped_filtered
    
    def flow_rigid(self, T_w0, T_w1):
        end_inds = self.warp_rigid(T_w0, T_w1)
        start_inds = self.events[self.warped_events_mask]
        
        # return just the event coordinates
        return start_inds[:, :2], end_inds[:, :2]
    
    def flow3D_rigid(self, T_w0, T_w1):
        start_coords, end_coords = self.flow_rigid(T_w0, T_w1)
        flow3D = coordinates_flow3D(start_coords, end_coords, self.image_shape)
        return flow3D
