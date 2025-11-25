import numpy as np
from ev_loader.RigidFlow import RigidFlow
from ev_loader.utils.utils_matrix_op import coordinates_flow3D, magnitude_of, coordinates_to_image
from ev_loader.utils.utils_losses import ase_loss, magnitude_difference_01_loss, mag_diff_loss

class ev_loader():
    def __init__(self, K, image_shape):
        self.K = K
        self.image_shape = image_shape
        self.median_lambda = 10
        self.prev_object_mask = None

    # Suppress based on magnitude difference
    def suppress(self, rigid_start_coords, rigid_end_coords, percived_flow):
        """Suppress just using the magnitude of the optical flow vectors
        """
        percived_start_coords, percived_end_coords = self.get_start_end_coords(percived_flow)
        # Compute flows magnitudes
        rigid_magnitude = magnitude_of(rigid_start_coords, rigid_end_coords)
        percived_magnitude = magnitude_of(percived_start_coords, percived_end_coords)

        # Cast coordinates to integers
        rigid_start_coords = rigid_start_coords.astype(int)
        percived_start_coords = percived_start_coords.astype(int)

        # Vectorized version of getting the segmentation mask
        rigid_mag_img = coordinates_to_image(rigid_start_coords, rigid_magnitude, self.image_shape)
        percived_mag_img = coordinates_to_image(percived_start_coords, percived_magnitude, self.image_shape)
        mask_rigid = rigid_mag_img != 0
        mask_perc = percived_mag_img != 0
        mask_both = mask_perc & mask_rigid
        
        # Compute the Loss
        thresholded_diff_magnitude, diff_mag = magnitude_difference_01_loss(
                                                percived_magnitude=percived_mag_img, 
                                                rigid_magnitude=rigid_mag_img, 
                                                validity_mask=mask_both,
                                                median_lambda=self.median_lambda
                                                )

        self.prev_object_mask = thresholded_diff_magnitude
        self.diff_mag = diff_mag
        return thresholded_diff_magnitude
    
    # Suppress based on angular difference
    def suppress_(self, rigid_start_coords, rigid_end_coords, percived_flow):
        # comupte the cosine similarity
        cosine_similarity = self.cosine_similarity_loss(rigid_start_coords, rigid_end_coords, percived_flow)

        # threshold the cosine similarity
        thresholded_cosine_similarity = cosine_similarity > 0.5
        return thresholded_cosine_similarity
    
    def cosine_similarity_loss(self, rigid_start_coords, rigid_end_coords, percived_flow):        
        rigid_flow = coordinates_flow3D(rigid_start_coords, rigid_end_coords)

        # comupte the cosine similarity
        cosine_similarity = self.cosine_similarity_2D(rigid_flow, percived_flow)
        return cosine_similarity

    def cosine_similarity_2D(self, A, B):
        # Compute cosine similarity between corresponding vectors
        dot_product = np.sum(A * B, axis=0)  # Element-wise dot product along the first axis (vector dimension)
        magnitude_A = np.sqrt(np.sum(A ** 2, axis=0))  # Magnitude of vectors in A
        magnitude_B = np.sqrt(np.sum(B ** 2, axis=0))  # Magnitude of vectors in B

        # Avoid division by zero
        cosine_similarity = np.divide(dot_product, magnitude_A * magnitude_B, where=(magnitude_A * magnitude_B) != 0)
        return cosine_similarity
    
    # Use L2 loss to compare the distance between two vector fields
    def suppress__(self, rigid_start_coords, rigid_end_coords, percived_flow, events):
        rigid_flow = coordinates_flow3D(rigid_start_coords, rigid_end_coords, image_shape=self.image_shape)

        # the loss should be defined only where events exists
        ase = ase_loss(percived_flow, rigid_flow)
        mag_diff = mag_diff_loss(percived_flow, rigid_flow)

        events_3D = coordinates_to_image(events[:,:2].astype(int), np.ones_like(events)[:,0], self.image_shape)
        ase_ = ase * events_3D
        mag_diff_ = mag_diff * events_3D
        ase_[200:, :] = np.min(ase_)
        mag_diff_[200:, :] = np.min(mag_diff_)

        #thresholded consensus loss
        thresh_ase_mag_dif = (ase_ > np.median(ase_[ase_ > 0])) & (mag_diff_ > np.median(mag_diff_[mag_diff_ > 0]))
        return thresh_ase_mag_dif
    
    @staticmethod
    def get_start_end_coords(percived_flow):
        y, x = np.mgrid[:percived_flow.shape[1], :percived_flow.shape[2]]

        gt_start_coords = np.stack((x, y), axis=0).reshape(2, -1).T
        gt_end_coords = gt_start_coords + percived_flow.reshape(2, -1).T

        mask = np.isnan(gt_end_coords).any(axis=1)
        gt_start_coords = gt_start_coords[~mask]
        gt_end_coords = gt_end_coords[~mask]
        return gt_start_coords, gt_end_coords
    

    def __call__(self, events, depth, T_w0, T_w1, percived_flow):
        # Compute rigid flow
        rigid_flow_ = RigidFlow(events=events, depth=depth, K=self.K, image_shape=self.image_shape)
        rigid_start_coords, rigid_end_coords = rigid_flow_.flow_rigid(T_w0=T_w0, T_w1=T_w1)

        return self.suppress(
            rigid_start_coords=rigid_start_coords, 
            rigid_end_coords=rigid_end_coords, 
            percived_flow=percived_flow
            )
        # return self.suppress__(
        #     rigid_start_coords=rigid_start_coords, 
        #     rigid_end_coords=rigid_end_coords, 
        #     percived_flow=percived_flow,
        #     events=events
        #     )