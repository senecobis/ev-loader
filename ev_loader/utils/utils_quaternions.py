import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_conjugate(quaternion):
    """
    Compute the conjugate of a quaternion.
    Args:
        quaternion (np.ndarray): Quaternion [w, x, y, z].
    Returns:
        np.ndarray: Conjugated quaternion [w, -x, -y, -z].
    """
    w, x, y, z = quaternion
    return np.array([w, -x, -y, -z])

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    Args:
        q1 (np.ndarray): First quaternion [w, x, y, z].
        q2 (np.ndarray): Second quaternion [w, x, y, z].
    Returns:
        np.ndarray: Resulting quaternion [w, x, y, z].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,  # w
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  # x
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,  # y
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2   # z
    ])

def relative_pose(quaternion_a, translation_a, quaternion_b, translation_b):
    """
    Compute the relative pose (rotation and translation) between two absolute poses.
    Args:
        quaternion_a (np.ndarray): Quaternion of pose A [w, x, y, z].
        translation_a (np.ndarray): Translation of pose A [x, y, z].
        quaternion_b (np.ndarray): Quaternion of pose B [w, x, y, z].
        translation_b (np.ndarray): Translation of pose B [x, y, z].
    Returns:
        tuple: Relative quaternion [w, x, y, z] and relative translation [x, y, z].
    """
    # Compute relative rotation
    q_a_inv = quaternion_conjugate(quaternion_a)
    q_rel = quaternion_multiply(quaternion_b, q_a_inv)

    # Rotate translation_a by q_rel
    t_a_quaternion = np.array([0, *translation_a])
    t_rotated = quaternion_multiply(quaternion_multiply(q_rel, t_a_quaternion), quaternion_conjugate(q_rel))[1:]

    # Compute relative translation
    t_rel = translation_b - t_rotated

    return q_rel, t_rel


def translation_and_quaternion_to_pose(translation, quaternion):
    """
    Convert a translation vector and quaternion to a 4x4 pose matrix.
    
    Args:
        translation (np.ndarray): Translation vector [tx, ty, tz].
        quaternion (np.ndarray): Quaternion [x, y, z, w].
        
    Returns:
        np.ndarray: 4x4 Pose matrix.
    """
    # Convert quaternion to rotation matrix
    rotation = R.from_quat(quaternion)  # [x, y, z, w]
    rotation_matrix = rotation.as_matrix()  # 3x3 rotation matrix

    # Create a 4x4 identity matrix
    pose_matrix = np.eye(4)

    # Fill in the rotation matrix
    pose_matrix[:3, :3] = rotation_matrix

    # Fill in the translation vector
    pose_matrix[:3, 3] = translation

    return pose_matrix


def poses_array_to_transformation_matrix(poses_array):
    T = []
    for pose in poses_array:
        translation = pose[:3]
        quaternion = pose[3:] # with format [x, y, z, w]
        T_ = translation_and_quaternion_to_pose(translation, quaternion)
        T.append(T_)
    return np.array(T)

