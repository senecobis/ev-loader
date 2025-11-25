import torch
import numpy as np
from matplotlib import pyplot as plt
from evlicious import Events
from pathlib import Path


def events_from_events(events, height=480, width=640):
    mask_x = (events[:, 0] > 0) & (events[:, 0] < width)
    mask_y = (events[:, 1] > 0) & (events[:, 1] < height)
    mask_events = mask_x & mask_y  # Use '&' instead of 'and'

    events_ = Events(
        x=events[:, 0][mask_events].astype(np.int16).astype(np.uint16),
        y=events[:, 1][mask_events].astype(np.int16).astype(np.uint16),
        t=events[:, 2][mask_events].astype(np.int64),
        p=events[:, 3][mask_events].astype(np.int8),
        width=width,
        height=height
    )
    return events_

def plot_prediction_and_gt(pred_t1, gt_t1, plot_path, ind, apply_sigmoid=True):
    plt.clf()
    assert len(pred_t1.shape) == 3
    assert len(gt_t1.shape) == 3

    pred_t1 = pred_t1.squeeze(0)
    gt_t1 = gt_t1.squeeze(0)

    if apply_sigmoid:
        pred_t1 = torch.sigmoid(pred_t1)

    pred_t1 = pred_t1 > 0.5

    # Plot and save prediction
    pred_fig, pred_ax = plt.subplots(figsize=(8, 8))
    pred_ax.imshow(pred_t1.cpu().numpy())
    pred_ax.axis("off")

    pred_out_path = plot_path / "Prediction"
    pred_out_path.mkdir(parents=True, exist_ok=True)
    pred_fig.savefig(pred_out_path / f"{str(ind).zfill(6)}.png")
    plt.close(pred_fig)

    # Plot and save ground truth
    gt_fig, gt_ax = plt.subplots(figsize=(8, 8))
    gt_ax.imshow(gt_t1.cpu().numpy())
    gt_ax.axis("off")

    gt_out_path = plot_path / "GT"
    gt_out_path.mkdir(parents=True, exist_ok=True)
    gt_fig.savefig(gt_out_path / f"{str(ind).zfill(6)}.png")
    plt.close(gt_fig)

def plot_all_events(events, plot_path, ind):
    plt.clf()
    events_ = events_from_events(events)
    rendered = events_.render()

    plt.figure(figsize=(12, 10))
    plt.imshow(rendered)
    plt.axis("off")

    out_path = plot_path / "Events"
    out_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path / f"{str(ind).zfill(6)}.png")
    plt.close()

def plot_all_events_only(events, plot_path, ind):
    plt.clf()
    events_ = events_from_events(events)

    x = events_.x.astype(np.int32)
    y = events_.y.astype(np.int32)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.invert_yaxis()  # Match image coordinate convention

    # Plot all events in light grey
    ax.scatter(x, y, s=1, color="lightgrey", alpha=0.6, label="Events")

    ax.axis("off")

    out_path = plot_path / "AllEvents"
    out_path.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_path / f"{str(ind).zfill(6)}.png")
    plt.close()

def plot_filtered_events(events, dynamic_mask, plot_path, ind):
    plt.clf()
    events_ = events_from_events(events)

    x = events_.x.astype(np.int32)
    y = events_.y.astype(np.int32)
    event_mask = dynamic_mask[y, x] == 1

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.invert_yaxis()

    # Plot all events in light grey
    ax.scatter(x, y, s=1, color="lightgrey", alpha=0.4, label="Original Events")

    # Plot filtered events in green
    ax.scatter(x[event_mask], y[event_mask], s=2, color="green", label="Filtered Events")

    ax.axis("off")

    out_path = plot_path / "FilteredEvents"
    out_path.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path / f"{str(ind).zfill(6)}.png")
    plt.close()

def plot_flow_arrows(flow, step=10, scale=1, save_path=None, dpi=300):
    """
    Plots optical flow as arrows using a quiver plot.

    Args:
        flow (np.ndarray or torch.Tensor): Optical flow of shape (H, W, 2).
        step (int): Sampling stride for arrows (larger = fewer arrows).
        scale (float): Scaling factor for arrows.
        save_path (str or Path): Where to save the figure. If None, it will display.
        dpi (int): Resolution for saving the figure.
    """
    if not isinstance(flow, np.ndarray):
        flow = flow.detach().cpu().permute(1, 2, 0).numpy()  # (H, W, 2)

    H, W, _ = flow.shape
    flow_u = flow[..., 0]
    flow_v = flow[..., 1]

    # Create a grid of coordinates
    x = np.arange(0, W, step)
    y = np.arange(0, H, step)
    X, Y = np.meshgrid(x, y)

    U = flow_v[::step, ::step]
    V = flow_u[::step, ::step]

    plt.figure(figsize=(8, 6), dpi=dpi)
    plt.quiver(X, Y, U, V, color='red', scale_units='xy', angles='xy', scale=scale)
    plt.gca().invert_yaxis()  # Match image coordinates
    plt.axis('off')
    plt.tight_layout(pad=0)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()
    else:
        plt.show()

def plot_flow_arrows_colored(flow, plot_path, ind, step=10, scale=1, cmap='viridis'):
    """
    Plots optical flow as arrows with color encoding for magnitude.

    Args:
        flow (np.ndarray or torch.Tensor): Optical flow of shape (H, W, 2).
        step (int): Sampling stride for arrows (larger = fewer arrows).
        scale (float): Scaling factor for arrows.
        cmap (str): Matplotlib colormap for magnitude coloring.
        save_path (str or Path): Where to save the figure. If None, it will display.
        dpi (int): Resolution for saving the figure.
    """
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    assert len(flow.shape) == 3  # (2, H, W)
    
    # Convert to numpy (H, W, 2)
    flow_np = flow.permute(1, 2, 0).detach().cpu().numpy()
    H, W, _ = flow_np.shape

    u = flow_np[..., 0]
    v = flow_np[..., 1]

    # Downsampled grid
    x = np.arange(0, W, step)
    y = np.arange(0, H, step)
    X, Y = np.meshgrid(x, y)

    U = v[::step, ::step]
    V = u[::step, ::step]

    # Magnitude for coloring
    magnitude = np.sqrt(U**2 + V**2).flatten()

    # Flatten for quiver
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    U_flat = U.flatten()
    V_flat = V.flatten()

    # Plot vector field with color based on magnitude
    q = ax.quiver(X_flat, Y_flat, U_flat, V_flat, magnitude, cmap=cmap,
                scale_units='xy', angles='xy', scale=scale)
    ax.invert_yaxis()
    ax.axis("off")
    # ax.set_title("Predicted optical flow")
    # fig.colorbar(q, ax=ax, label='Flow magnitude')

    plt.tight_layout()

    # Ensure output directory exists
    plot_path = Path(plot_path) / "flow"
    plot_path.mkdir(parents=True, exist_ok=True)

    # Save plot
    plt.savefig(plot_path / f"{str(ind).zfill(6)}.png", bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)

def plot_events_with_flow(events, flow, plot_path, ind, step=10, scale=1, cmap='viridis'):
    """
    Plots events as background and overlays optical flow arrows.

    Args:
        events: Object containing x and y event coordinates (and optionally timestamps or polarities).
        flow (torch.Tensor): Optical flow tensor of shape (2, H, W).
        plot_path (str or Path): Directory where to save the output image.
        ind (int): Index for filename.
        step (int): Sampling stride for arrows.
        scale (float): Scaling factor for arrows.
        cmap (str): Colormap used for magnitude coloring.
    """
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.invert_yaxis()
    ax.axis("off")

    # === Events ===
    events_ = events_from_events(events)
    x = events_.x.astype(np.int32)
    y = events_.y.astype(np.int32)
    ax.scatter(x, y, s=1, color="lightgrey", label="Events")

    # === Optical Flow ===
    assert len(flow.shape) == 3  # (2, H, W)
    flow_np = flow.permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 2)
    H, W, _ = flow_np.shape

    u = flow_np[..., 0]
    v = flow_np[..., 1]

    # Downsample for visualization
    x_grid = np.arange(0, W, step)
    y_grid = np.arange(0, H, step)
    X, Y = np.meshgrid(x_grid, y_grid)

    U = v[::step, ::step]
    V = u[::step, ::step]
    magnitude = np.sqrt(U**2 + V**2).flatten()

    # Flatten for quiver
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    U_flat = U.flatten()
    V_flat = V.flatten()

    # Plot arrows over event image
    ax.quiver(X_flat, Y_flat, U_flat, V_flat, magnitude,
              cmap=cmap, scale_units='xy', angles='xy', scale=scale, alpha=0.5)

    # Save
    save_dir = Path(plot_path) / "EventsWithFlow"
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_dir / f"{str(ind).zfill(6)}.png", bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()