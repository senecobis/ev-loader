import numpy as np


def epe_loss(flow1, flow2, epsilon=1e-6):
    """Relative Endpoint Error (Relative EPE)"""
    # Compute components of flow vectors
    u1, v1 = flow1[0], flow1[1]
    u2, v2 = flow2[0], flow2[1]

    # Magnitudes of flow vectors
    mag2 = np.sqrt(u2**2 + v2**2)

    # Relative Endpoint Error (Relative EPE)
    endpoint_error = np.sqrt((u1 - u2)**2 + (v1 - v2)**2)
    relative_epe = endpoint_error / (mag2 + epsilon)
    return relative_epe


def ase_loss(flow1, flow2, epsilon=1e-6):
    """Angular Similarity Error (ASE)"""
    # Compute components of flow vectors
    u1, v1 = flow1[0], flow1[1]
    u2, v2 = flow2[0], flow2[1]

    # Magnitudes of flow vectors
    mag1 = np.sqrt(u1**2 + v1**2)
    mag2 = np.sqrt(u2**2 + v2**2)

    # Angular Similarity Error (ASE)
    dot_product = u1 * u2 + v1 * v2
    cosine_similarity = dot_product / ((mag1 * mag2) + epsilon)
    angular_similarity_error = 1 - cosine_similarity
    return angular_similarity_error


def epe_ase_loss(flow1, flow2, alpha=0.5, beta=0.5, epsilon=1e-6):
    """
    Computes the combined loss between two optical flow fields.

    Parameters:
    - flow1: np.ndarray of shape (2, H, W) representing the first flow field.
    - flow2: np.ndarray of shape (2, H, W) representing the second flow field.
    - alpha: float, weight for Relative Endpoint Error.
    - beta: float, weight for Angular Similarity Error.
    - epsilon: small float to avoid division by zero.

    Returns:
    - combined_loss: np.ndarray of shape (H, W) representing the per-pixel combined loss.
    """
    # Compute components of flow vectors
    u1, v1 = flow1[0], flow1[1]
    u2, v2 = flow2[0], flow2[1]

    # Magnitudes of flow vectors
    mag1 = np.sqrt(u1**2 + v1**2)
    mag2 = np.sqrt(u2**2 + v2**2)

    # Relative Endpoint Error (Relative EPE)
    endpoint_error = np.sqrt((u1 - u2)**2 + (v1 - v2)**2)
    relative_epe = endpoint_error / (mag2 + epsilon)

    # Angular Similarity Error (ASE)
    dot_product = u1 * u2 + v1 * v2
    cosine_similarity = dot_product / ((mag1 * mag2) + epsilon)
    angular_similarity_error = 1 - np.clip(cosine_similarity, -1, 1)

    # Combined loss
    combined_loss = alpha * relative_epe + beta * angular_similarity_error

    return combined_loss


def mag_diff_loss(flow1, flow2):
    """Magnitude Difference 01 Loss"""
    u1, v1 = flow1[0], flow1[1]
    u2, v2 = flow2[0], flow2[1]

    # Magnitudes of flow vectors
    mag1 = np.sqrt(u1**2 + v1**2)
    mag2 = np.sqrt(u2**2 + v2**2)

    diff = np.abs(mag1 - mag2)
    return diff


def magnitude_difference_01_loss(percived_magnitude, rigid_magnitude, validity_mask, median_lambda):
    """ This function computes the difference in magnitudes between the percived and rigid flow 
    where both of the 2 exists (validity mask). Then is thresholds the difference to get the independently moving objects

    Args:
        percived_magnitude (np.ndarray): The magnitude of the percived flow
        rigid_magnitude (np.ndarray): the magnitude of the rigid flow
        validity_mask (np.ndarray): The mask where both of the flows exist
        median_lambda (float): The lambda value to multiply the median with

    Returns:
        np.ndarray: The binary mask representing the thresholded difference in magnitudes
    """
    # Compute the difference in magnitudes
    diff_magnitudes_image = np.abs(percived_magnitude - rigid_magnitude)

    # Threshold the indipendently moving objects
    # TODO How I decided the threshold is based on median (for now)
    median = np.median(diff_magnitudes_image[validity_mask])

    # if the median is very low there is no motion
    if median < 0.1:
        return np.zeros_like(diff_magnitudes_image)
    
    thresholded_diff_magnitude = diff_magnitudes_image > median_lambda*median
    return thresholded_diff_magnitude, diff_magnitudes_image


def loss_depth_weighted_flow_magnitude_difference(percived_mag_img, rigid_mag_img, validity_mask, depth, median_lambda):
    """ This function computes the difference in magnitudes between the percived and rigid flow 
    where both of the 2 exists (validity mask). Then is thresholds the difference to get the independently moving objects

    Args:
        percived_mag_img (np.ndarray): The magnitude of the percived flow
        rigid_mag_img (np.ndarray): the magnitude of the rigid flow
        validity_mask (np.ndarray): The mask where both of the flows exist

    Returns:
        np.ndarray: The binary mask representing the thresholded difference in magnitudes
    """
    # Compute the difference in magnitudes
    diff_magnitudes_image = np.abs(percived_mag_img - rigid_mag_img) * depth

    # Threshold the indipendently moving objects
    # TODO How I decided the threshold is based on median (for now)
    median = np.median(diff_magnitudes_image[validity_mask])

    # if the median is very low there is no motion
    if median < 0.1:
        return np.zeros_like(diff_magnitudes_image)
    
    thresholded_diff_magnitude = diff_magnitudes_image > median_lambda*median
    return thresholded_diff_magnitude
