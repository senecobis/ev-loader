import numpy as np
from skimage.measure import label
import torch

def object_success_proxy(gt_bin, pred_bin, iou_thresh=0.5, connectivity=2):
    """
    Coverage-based proxy: each GT component is 'correct' if >= cov_thresh
    of its pixels are predicted positive. Does NOT penalize merges.
    Returns: success_count, gt_count, ratio
    """
    gt_lab = label(gt_bin.astype(bool), connectivity=connectivity)
    gt_ids = np.unique(gt_lab); gt_ids = gt_ids[gt_ids != 0]
    if len(gt_ids) == 0:
        return 0, 0, np.nan

    P = pred_bin.astype(bool)
    success = 0
    for gid in gt_ids:
        G = (gt_lab == gid)
        inter = np.count_nonzero(G & P)
        G_area = np.count_nonzero(G)
        coverage = inter / G_area if G_area > 0 else 0.0
        success += (coverage >= iou_thresh)
    return success, len(gt_ids), success / len(gt_ids)

def instance_success_from_binary(gt_bin, pred_bin, iou_thresh=0.5, connectivity=2):
    """
    Ratio (and counts) of correctly segmented objects from binary masks.
    - gt_bin, pred_bin: HxW arrays of {0,1} (or booleans)
    - connectivity: 1 (4-neigh) or 2 (8-neigh)
    Returns: success_count, gt_count, ratio
    """
    if isinstance(gt_bin, torch.Tensor):
        gt_bin = gt_bin.cpu().numpy()
    if isinstance(pred_bin, torch.Tensor):
        pred_bin = pred_bin.cpu().numpy()
    
    gt_lab  = label(gt_bin.astype(bool), connectivity=connectivity)   # 0=bg, 1..K
    pr_lab  = label(pred_bin.astype(bool), connectivity=connectivity) # 0=bg, 1..M
    gt_ids  = np.unique(gt_lab);  gt_ids  = gt_ids[gt_ids != 0]
    pr_ids  = np.unique(pr_lab);  pr_ids  = pr_ids[pr_ids != 0]

    K = len(gt_ids)
    if K == 0:  # no GT objects in this image
        return 0, 0, np.nan

    # Precompute areas
    gt_areas = {gid: np.count_nonzero(gt_lab == gid) for gid in gt_ids}
    pr_areas = {pid: np.count_nonzero(pr_lab == pid) for pid in pr_ids}

    success = 0
    for gid in gt_ids:
        G = (gt_lab == gid)
        best_iou = 0.0
        for pid in pr_ids:
            P = (pr_lab == pid)
            inter = np.count_nonzero(G & P)
            if inter == 0: 
                continue
            union = gt_areas[gid] + pr_areas[pid] - inter
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
        if best_iou >= iou_thresh:
            success += 1

    return success, K, success / K


if __name__ == "__main__":
    # Example usage
    gt_example = np.array([[0, 1, 1, 0, 0],
                           [0, 1, 0, 0, 1],
                           [0, 0, 0, 1, 1],
                           [1, 1, 0, 0, 0]])
    
    pred_example = np.array([[0, 1, 1, 0, 0],
                             [0, 1, 0, 0, 1],
                             [0, 0, 1, 0, 1],
                             [0, 1, 0, 0, 1]])

    s, k, r = object_success_proxy(
        gt_example, pred_example, iou_thresh=0.5
        )
    print(f"Object Success@0.5: {s}/{k} = {100*r:.1f}%")
    
    s_inst, k_inst, r_inst = instance_success_from_binary(
        gt_example, pred_example, iou_thresh=0.5
        )
    print(f"Instance Success@0.5: {s_inst}/{k_inst} = {100*r_inst:.1f}%")