import torch
import numpy as np
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class LinearMaskExtrapolator:
    def __init__(self):
        self.dynamic_mask_t0 = None
    
    @staticmethod
    def get_centroids(binary_mask: torch.Tensor):
        """Find the centroids of the connected components in a binary mask
        and return the labeled mask.
        Args:
            binary_mask (torch.Tensor): A binary mask of shape (H, W) with 1s for the object
        Returns:
            np.ndarray: centroids (2, N_centroids) 
            np.ndarray: labeled_mask (H, W) with labels for each connected component
        """
        # Ensure it's a CPU numpy array
        mask_np = binary_mask.cpu().numpy().astype(np.uint8)
        
        # Step 1: Label connected components
        labeled_array, num_features = label(mask_np) # labeled_array: shape (H, W), num_features: int

        # Step 2: Compute centroids
        centroids = center_of_mass(mask_np, labeled_array, range(1, num_features + 1))
        centroids_array = np.array(centroids).T  # shape: (2, N_centroids)

        return centroids_array, labeled_array
    
    @staticmethod
    def optimal_match_centroids(centroids_A, centroids_B):
        """Find the optimal matching between two sets of centroids using the Hungarian algorithm.
        Args:
            centroids_A (np.ndarray): Centroids from the first mask (2, N_centroids_A)
            centroids_B (np.ndarray): Centroids from the second mask (2, N_centroids_B)
        Returns:
            np.ndarray: Array of shape (N_matches, 2) where each row is a pair of matched indices
        """
        # Calculate the distance matrix between centroids
        # centroids_A: shape (2, N_centroids_A)
        # centroids_B: shape (2, N_centroids_B)
        # distances: shape (N_centroids_A, N_centroids_B)
        A = centroids_A.T
        B = centroids_B.T
        distances = cdist(A, B)

        row_ind, col_ind = linear_sum_assignment(distances)
        
        # The returned matches is a 1D mapping where for the Nth centroid in A,
        # Identified as row_ind[N], the corresponding centroid in B is col_ind[N].
        return np.array(list(zip(row_ind, col_ind)))
    
    @staticmethod
    def extrapolated_centroids(match_, centroids_t0, centroids_t1):
        # match_[0] should take the centroid number associated with centroid number match_[1]
        coords_cluster = np.stack((centroids_t0[:, match_[0]], centroids_t1[:, match_[1]]), axis=0)
        t = [0, 1]
        coeffs_x = np.polyfit(t, coords_cluster[:, 0], deg=1)
        coeffs_y = np.polyfit(t, coords_cluster[:, 1], deg=1)
        extrapolated_x = np.polyval(coeffs_x, x=2)
        extrapolated_y = np.polyval(coeffs_y, x=2)
        
        # the future centroid position
        return np.array([extrapolated_x, extrapolated_y])

    @staticmethod
    def shifted_coords_(coords, flow, H, W):
        # coords are (row, col) | flow is (y, x) = (row, col)
        shifted_coords = coords + flow
        # Round and clip to valid indices
        shifted_coords = np.round(shifted_coords).astype(int)
        valid = (0 <= shifted_coords[:, 0]) & (shifted_coords[:, 0] < H) & \
                (0 <= shifted_coords[:, 1]) & (shifted_coords[:, 1] < W)
        return shifted_coords[valid]

    @staticmethod
    def extrapolate_dynamic_mask(centroids_t0, centroids_t1, matches, labled_mask_t1):
        future_mask = np.zeros_like(labled_mask_t1)
        for match_ in matches:
            # the future centroid position
            x_y_t0 = centroids_t0[:, match_[0]]
            x_y_t1 = centroids_t1[:, match_[1]]
            x_y_t2 = LinearMaskExtrapolator.extrapolated_centroids(match_, centroids_t0, centroids_t1)
            flow = x_y_t2 - x_y_t1

            cluster_label = match_[0] + 1
            cluster_mask = (labled_mask_t1 == cluster_label)
            coords = np.argwhere(cluster_mask)

            # Shift each coordinate
            H, W = future_mask.shape
            shifted_coords = LinearMaskExtrapolator.shifted_coords_(coords, flow, H, W)
            future_mask[shifted_coords[:, 0], shifted_coords[:, 1]] = 1
        
        return future_mask

    @staticmethod
    def has_an_object(mask):
        return (mask == True).any().item()
    
    def reset(self):
        self.dynamic_mask_t0 = None
        
    def __call__(self, pred_logits):
        """call the linear extrapolator

        Args:
            pred_logits (torch.Tensor): H X W sized tensor with predicted logits from the model

        Returns:
            torch.tensor: the extrapolated mask
        """
        assert len(pred_logits.shape) == 2, "pred_logits should be a 2D tensor"
        dynamic_mask_t1 = (torch.sigmoid(pred_logits) > 0.5).cpu()

        if self.dynamic_mask_t0 is None:
            print("No previous mask, using the current one")
            self.dynamic_mask_t0 = dynamic_mask_t1
            return dynamic_mask_t1.to(pred_logits.device)

        # calculate the centroids ONLY if the masks have objects
        if self.has_an_object(self.dynamic_mask_t0) and self.has_an_object(dynamic_mask_t1):
            centroids_t0, labeld_mask_t0 = self.get_centroids(self.dynamic_mask_t0)
            centroids_t1, labeld_mask_t1 = self.get_centroids(dynamic_mask_t1)
            matches = self.optimal_match_centroids(centroids_t0, centroids_t1)
            future_mask = self.extrapolate_dynamic_mask(
                centroids_t0, centroids_t1, matches, labeld_mask_t1
                )
            future_mask = torch.tensor(future_mask)
        else:
            future_mask = dynamic_mask_t1
        
        # Save current mask for the next iter
        self.dynamic_mask_t0 = dynamic_mask_t1
        return future_mask.to(pred_logits.device)
    

if __name__ == "__main__":
    import torch
    import numpy as np
    from scipy.ndimage import label, center_of_mass
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

    # Example usage
    pred_logits = torch.randn(100, 100)  # Example logits
    extrapolator = LinearMaskExtrapolator()
    future_mask = extrapolator(pred_logits)
    
    pred_logits_2 = torch.randn(100, 100)  # Example logits
    future_mask_2 = extrapolator(pred_logits_2)
    print(future_mask_2.shape)