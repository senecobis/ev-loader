import torch
import numpy as np
from .LinearMaskExtrapolator import LinearMaskExtrapolator
from scipy.ndimage import center_of_mass


class MaskFlow(LinearMaskExtrapolator):
    def __init__(self):
        super().__init__()
        self.dynamic_mask_t0 = None
    
    @staticmethod
    def masks_flow(centroids_t0, centroids_t1, matches):
        """Calculate the flow of the centroids between two masks.
        Args:
            centroids_t0 (np.ndarray): Centroids from the first mask (2, N_centroids_A)
            centroids_t1 (np.ndarray): Centroids from the second mask (2, N_centroids_B)
            matches (np.ndarray): Array of shape (N_matches, 2) where each row is a pair of matched indices
        Returns:
            np.ndarray: Array of shape (N_matches, 2) where each row is the flow vector for the matched centroids
        """
        flows = {}
        for match_ in matches:
            x_y_t0 = centroids_t0[:, match_[0]]
            x_y_t1 = centroids_t1[:, match_[1]]
            flows[match_[0]] = MaskFlow.mask_flow(x_y_t0, x_y_t1)
        return flows
    
    @staticmethod
    def mask_flow(x_y_t0, x_y_t1):
        flow = {}
        flow["flow"] = x_y_t1 - x_y_t0
        flow["centroid_xy_t0"] = x_y_t0
        flow["centroid_xy_t1"] = x_y_t1
        return flow
    
    @staticmethod
    def has_an_object(mask):
        return (mask > 0).any()
    
    @staticmethod
    def get_centroids_from_labled_masks(mask):
        mask = mask.astype(int)
        labels = np.unique(mask)
        labels = labels[labels != 0]  # Remove background label if it's 0
        centroids = {}

        for label in labels:
            centroid = center_of_mass(mask == label)
            centroids[label] = np.array(centroid)  # (row, column) or (y, x)

        return centroids, labels
    
    def flow_calculation_with_lables(self, dynamic_mask_t1):
        """Calculate the flow of the centroids between two masks.
        Args:
            dynamic_mask_t1 (torch.Tensor): The current mask (H, W)
        Returns:
            dict: A dictionary containing the flow vectors for each matched centroid
        """
        assert len(dynamic_mask_t1.shape) == 2, "dynamic_mask_t1 should be a 2D tensor"
        
        if self.dynamic_mask_t0 is None:
            self.dynamic_mask_t0 = dynamic_mask_t1
            # No previous mask, using the current one
            return None
        if self.has_an_object(self.dynamic_mask_t0) and self.has_an_object(dynamic_mask_t1):
            centroids_t0, labeled_mask_t0 = self.get_centroids_from_labled_masks(self.dynamic_mask_t0)
            centroids_t1, labeled_mask_t1 = self.get_centroids_from_labled_masks(dynamic_mask_t1)
            
            if not set(labeled_mask_t1).issuperset(labeled_mask_t0):
                self.dynamic_mask_t0 = dynamic_mask_t1
                print("Some objects went out the scene")
                return None
            
            labeled_flows = {}
            for label in centroids_t0:
                centr_t0 = centroids_t0[label]
                centr_t1 = centroids_t1[label]
                labeled_flows[label] = self.mask_flow(centr_t0, centr_t1)
            self.dynamic_mask_t0 = dynamic_mask_t1
            return labeled_flows
        else:
            # No objects
            self.dynamic_mask_t0 = dynamic_mask_t1
            return None

    
    def flow_calculation(self, dynamic_mask_t1: torch.Tensor):
        """Calculate the flow of the centroids between two masks.
        Args:
            dynamic_mask_t1 (torch.Tensor): The current mask (H, W)
        Returns:
            dict: A dictionary containing the flow vectors for each matched centroid
        """
        assert len(dynamic_mask_t1.shape) == 2, "dynamic_mask_t1 should be a 2D tensor"
        assert isinstance(dynamic_mask_t1, torch.Tensor), "dynamic_mask_t1 should be a torch tensor"
        
        if self.dynamic_mask_t0 is None:
            self.dynamic_mask_t0 = dynamic_mask_t1
            print("No previous mask, using the current one")
            return None
        if self.has_an_object(self.dynamic_mask_t0) and self.has_an_object(dynamic_mask_t1):
            centroids_t0, labeld_mask_t0 = self.get_centroids(self.dynamic_mask_t0)
            centroids_t1, labeld_mask_t1 = self.get_centroids(dynamic_mask_t1)
            matches = self.optimal_match_centroids(centroids_t0, centroids_t1)
            return self.masks_flow(centroids_t0, centroids_t1, matches)
        else:
            print("No objects")
            return None
        
        
from suppressor.utils.utils import plot_arrows

if __name__ == "__main__":
    # Example usage
    mask_flow = MaskFlow()
    dynamic_mask_t0 = torch.zeros((480, 640), dtype=torch.float32)
    dynamic_mask_t0[100:200, 100:200] = 1
    flow = mask_flow.flow_calculation(dynamic_mask_t0)
    
    dynamic_mask_t1 = torch.zeros((480, 640), dtype=torch.float32)
    dynamic_mask_t1[300:400, 400:600] = 1
    flow = mask_flow.flow_calculation(dynamic_mask_t1)
    
    start_coords = flow[0]["centroid_xy_t0"][np.newaxis, :]
    end_coords = flow[0]["centroid_xy_t1"][np.newaxis, :]
    
    # from xy to yx
    start_coords = start_coords[:, ::-1]
    end_coords = end_coords[:, ::-1]
    
    
    fig = plot_arrows(image=dynamic_mask_t0+dynamic_mask_t1, start_coords=start_coords, end_coords=end_coords)
    fig.savefig("test.png")
    print(flow)