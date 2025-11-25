import torch
import json
import numpy as np
import rerun as rr
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from ..data.events import Events, RenderingType


def render_events(events, image_shape):
    """
    Render events into an image.
    
    Args:
    events (np.ndarray): Array of shape (N, 4) containing [x, y, t, polarity]
    image_shape (tuple): Shape of the image
    
    Returns:
    np.ndarray: Rendered image
    """
    if isinstance(events, torch.Tensor):
        events = events.clone().cpu().numpy()
    
    events_ = Events(
                x=events[...,0].astype(np.uint16), 
                y=events[...,1].astype(np.uint16),
                t=events[...,2].astype(np.int64),
                p=events[...,3].astype(np.int8),
                width=image_shape[0], 
                height=image_shape[1]
                )
    rendered = events_.render()
    return rendered


def render_events_timesurface(events, image_shape):
    """
    Render events into a timesurface.
    
    Args:
    events (np.ndarray): Array of shape (N, 4) containing [x, y, t, polarity]
    image_shape (tuple): Shape of the image
    
    Returns:
    np.ndarray: Rendered image
    """
    events = events.clone().cpu().numpy()
    events_ = Events(
                x=events[...,0].astype(np.uint16), 
                y=events[...,1].astype(np.uint16),
                t=events[...,2].astype(np.int64),
                p=events[...,3].astype(np.int8),
                width=image_shape[0], 
                height=image_shape[1]
                )
    rendered = events_.render(rendering_type=RenderingType.TIME_SURFACE)
    return rendered


def plot_arrows(image, start_coords, end_coords, width=2, scale=1, alpha=0.5):
    """
    Draw 2D vectors on an image using matplotlib with colors based on vector magnitudes.

    Parameters:
    - image: 2D numpy array representing the image
    - start_coords: Nx2 array of starting coordinates (y, x)
    - end_coords: Nx2 array of ending coordinates (y, x)
    - width: width of the vector lines (default: 2)
    - scale: scaling factor for vector size (default: 1)

    Returns:
    - Figure with the image and drawn vectors
    """
    
    # Create a new figure and display the image
    plt.clf()
    plt.close('all')  # Close any open figures to avoid duplicates

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.imshow(image, alpha=alpha)  # Display the underlying image

    if isinstance(start_coords, torch.Tensor):
        start_coords = start_coords.cpu().numpy()
    if isinstance(end_coords, torch.Tensor):
        end_coords = end_coords.cpu().numpy()

    # Calculate the vector components
    dx = end_coords[:, 0] - start_coords[:, 0]
    dy = end_coords[:, 1] - start_coords[:, 1]

    # Compute vector magnitudes
    magnitudes = np.sqrt(dx**2 + dy**2)

    # Normalize magnitudes for colormap (avoid division by zero)
    norm = plt.Normalize(vmin=np.min(magnitudes), vmax=np.max(magnitudes))
    cmap = plt.cm.viridis  # Use a colormap (e.g., viridis)
    
    # Map magnitudes to colors
    colors = cmap(norm(magnitudes))

    # Draw the vectors with colors based on magnitudes
    ax.quiver(start_coords[:, 0], start_coords[:, 1], dx, dy,
              angles='xy', scale_units='xy', scale=scale,
              color=colors, width=width / 1000)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a colorbar for reference
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation='vertical', label='Vector Magnitude')
    return fig


def plot_3Darrows(flow, image=None, subsample=1, save_path=None, no_gray=False):
    """ Plot the optical flow as arrows on top of the image.

    Args:
    flow (np.ndarray): Optical flow tensor of shape (2, H, W)
    image (np.ndarray): Image tensor of shape (H, W)
    subsample (int): Subsampling factor for the arrows
    save_path (str): Path to save the figure
    no_gray (bool): If True, the image will be displayed in color

    Returns:
    fig, ax: Figure and axis objects
    """
    plt.clf()

    if image is None:
        image = np.ones((flow.shape[1], flow.shape[2]))

    # Extract flow components
    x_magnitudes = flow[0, ...]  # Horizontal component
    y_magnitudes = flow[1, ...]  # Vertical component

    # Create a grid of pixel coordinates
    y, x = np.mgrid[:image.shape[0], :image.shape[1]]

    # Compute vector magnitudes
    magnitudes = np.sqrt(x_magnitudes**2 + y_magnitudes**2)

    # Mask to filter out zero magnitudes
    nonzero_mask = magnitudes > 0

    # Subsample the grid, flow components, and mask
    x = x[::subsample, ::subsample]
    y = y[::subsample, ::subsample]
    x_magnitudes = x_magnitudes[::subsample, ::subsample]
    y_magnitudes = y_magnitudes[::subsample, ::subsample]
    magnitudes = magnitudes[::subsample, ::subsample]
    nonzero_mask = nonzero_mask[::subsample, ::subsample]

    # Apply the nonzero mask
    x = x[nonzero_mask]
    y = y[nonzero_mask]
    x_magnitudes = x_magnitudes[nonzero_mask]
    y_magnitudes = y_magnitudes[nonzero_mask]
    magnitudes = magnitudes[nonzero_mask]

    # Normalize magnitudes for colormap
    norm = plt.Normalize(vmin=np.min(magnitudes), vmax=np.max(magnitudes))
    cmap = plt.cm.viridis  # Use a colormap (e.g., viridis)
    arrow_colors = cmap(norm(magnitudes))

    # Plot the image and arrows
    fig, ax = plt.subplots(figsize=(13, 8))
    if no_gray:
        ax.imshow(image, alpha=1)  # Display the underlying image
    else:
        ax.imshow(image, cmap='gray', alpha=1)  # Display the underlying image

    ax.quiver(
        x, y, x_magnitudes, y_magnitudes,
        angles='xy', scale_units='xy', scale=1,
        color=arrow_colors, width=0.003
    )

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a colorbar for reference
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation='vertical', label='Vector Magnitude')

    if save_path is not None:
        plt.savefig(save_path)
    return fig


def rerun_log_pose(entity_path, pose_matrix):
    rotation = pose_matrix[:3, :3]
    translation = pose_matrix[:3, 3]
    quaternion = Rotation.from_matrix(rotation).as_quat()

    rng = np.random.default_rng(12345)
    image = rng.uniform(0, 255, size=[240, 346, 3])
    
    rr.log(
        entity_path,
        rr.Transform3D(
            translation=translation,
            rotation=rr.Quaternion(xyzw=quaternion)
        ),
        rr.Pinhole(
            focal_length=100,
            # camera_xyz=rr.ViewCoordinates.RUB, 
            image_plane_distance=0.1,
            width=346,
            height=240,
        ),

        rr.Image(image)
    )


def rerun_log_pose_image(entity_path, pose_matrix, image):
    rotation = pose_matrix[:3, :3]
    translation = pose_matrix[:3, 3]
    quaternion = Rotation.from_matrix(rotation).as_quat()

    image_shape = list(image.shape)
    width = max(image_shape)
    height = min(image_shape)
    
    rr.log(
        entity_path,
        rr.Transform3D(
            translation=translation,
            rotation=rr.Quaternion(xyzw=quaternion)
        ),
        rr.Pinhole(
            focal_length=100,
            image_plane_distance=0.1,
            width=width,
            height=height,
        ),

        rr.Image(image)
    )


def rerun_log_pose_as_t_and_quaterions(entity_path, translation, quaternion, image=None):
    if image is None:
        rng = np.random.default_rng(12345)
        image = rng.uniform(0, 255, size=[240, 346, 3])
    height, width, _ = image.shape
    
    rr.log(
        entity_path,
        rr.Transform3D(
            translation=translation,
            rotation=rr.Quaternion(xyzw=quaternion)
        ),
        rr.Pinhole(
            focal_length=100,
            # camera_xyz=rr.ViewCoordinates.RUB, 
            image_plane_distance=0.1,
            width=width,
            height=height,
        ),

        rr.Image(image)
    )


def find_closest(data, timestamps, event_ts):
    time_differences = np.abs(timestamps - event_ts)
    closest_index = np.argmin(time_differences)
    return data[closest_index]


def get_device(gpu_num=0):
    """
    Get the device to use in the pipeline.
    """
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(gpu_num) if cuda else "cpu")
    return device


def open_config_json(config_path):
    config = json.load(open(config_path, 'r'))
    return config