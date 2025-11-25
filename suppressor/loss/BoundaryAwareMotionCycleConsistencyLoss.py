import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryAwareMotionCycleConsistencyLoss(nn.Module):
    def __init__(self, motion_thresh=1e-3, boundary_thresh=1e-3, eps=1e-6):
        super(BoundaryAwareMotionCycleConsistencyLoss, self).__init__()
        self.motion_thresh = motion_thresh
        self.boundary_thresh = boundary_thresh
        self.eps = eps

    def compute_boundary_map(self, seg_mask):
        # Simple sobel-like gradient to approximate edges
        grad_x = seg_mask[:, :, :, 1:] - seg_mask[:, :, :, :-1]
        grad_y = seg_mask[:, :, 1:, :] - seg_mask[:, :, :-1, :]
        
        # Pad to maintain original size
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        
        boundary = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        return boundary

    def forward(self, seg_mask, flow_fwd):
        """
        Args:
            seg_mask (torch.Tensor): Binary segmentation map, shape (B, 1, H, W)
            flow_fwd (torch.Tensor): Forward optical flow, shape (B, 2, H, W)
        Returns:
            loss (torch.Tensor): Boundary-aware motion cycle consistency loss
        """

        B, _, H, W = flow_fwd.shape

        # Compute boundary map
        boundary = self.compute_boundary_map(seg_mask)
        boundary_mask = (boundary > self.boundary_thresh).float()

        # Create a mesh grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=flow_fwd.device),
            torch.linspace(-1, 1, W, device=flow_fwd.device),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), 2)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)

        # Normalize flow
        flow_fwd_norm = torch.zeros_like(flow_fwd)
        flow_fwd_norm[:, 0, :, :] = flow_fwd[:, 0, :, :] / ((W - 1.0) / 2.0)
        flow_fwd_norm[:, 1, :, :] = flow_fwd[:, 1, :, :] / ((H - 1.0) / 2.0)

        # Warp forward
        grid_warped = grid + flow_fwd_norm.permute(0, 2, 3, 1)
        seg_mask_warped = F.grid_sample(seg_mask, grid_warped, align_corners=True, mode='bilinear', padding_mode='border')

        # Approximate backward flow
        flow_bwd_norm = -flow_fwd_norm

        # Warp back
        grid_warped_back = grid_warped + flow_bwd_norm.permute(0, 2, 3, 1)
        seg_mask_cycle = F.grid_sample(seg_mask_warped, grid_warped_back, align_corners=True, mode='bilinear', padding_mode='border')

        # Motion mask: only moving regions
        flow_mag = torch.norm(flow_fwd, dim=1, keepdim=True)
        motion_mask = (flow_mag > self.motion_thresh).float()

        # Combine masks: moving AND near boundary
        effective_mask = (boundary_mask * motion_mask)

        # Loss: only on effective masked regions
        diff = (seg_mask - seg_mask_cycle).abs()
        loss = (diff * effective_mask).sum() / (effective_mask.sum() + self.eps)

        return loss


if __name__ == "__main__":
    # Example usage
    B, C, H, W = 11, 5, 100, 100
    seg_mask = torch.randint(0, 2, (B, C, H, W)).float() 
    flow_fwd = torch.ones(B, 2, H, W)*10**1

    loss_fn = BoundaryAwareMotionCycleConsistencyLoss(
        motion_thresh=1e-3, 
        boundary_thresh=1e-3
        )
    # loss = loss_fn(seg_mask, flow_fwd)/torch.norm(flow_fwd, dim=1).mean()
    loss = loss_fn(seg_mask, flow_fwd)
    print("Motion Cycle Consistency Loss:", loss.item())
