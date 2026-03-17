import glob
import inspect
import os
import cv2
import numpy as np
import torch

from evlicious import Events

DSEC_CLASSES = ['pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train']
DSEC_COLORS = np.array(
    [
        [0.121, 0.466, 0.705],
        [1.000, 0.498, 0.054],
        [0.172, 0.627, 0.172],
        [0.839, 0.152, 0.156],
        [0.580, 0.404, 0.741],
        [0.549, 0.337, 0.294],
        [0.890, 0.467, 0.761],
        [0.498, 0.498, 0.498],
    ],
    dtype=np.float32,
)
GEN1_CLASSES = ['car', 'pedestrian']
GEN1_COLORS = np.array(
    [
        [0.121, 0.466, 0.705],
        [1.000, 0.498, 0.054],
    ],
    dtype=np.float32,
)


def render_events_on_image(image, x, y, p, t):
    viz_h, viz_w = image.shape[0], image.shape[1]
    mask = (x >= 0) & (x < viz_w) & (y >= 0) & (y < viz_h)
    events = Events(
        x=x[mask].astype(np.uint16),
        y=y[mask].astype(np.uint16),
        t=t[mask].astype(np.int64),
        p=p[mask].astype(np.int8),
        width=viz_w,
        height=viz_h,
    )
    return events.render(image)


def _draw_bbox_on_img(
    img,
    x,
    y,
    w,
    h,
    labels,
    scores=None,
    conf=0.5,
    label="",
    scale=1,
    linewidth=2,
    show_conf=True,
    dataset="dsec", # "gen1" or "dsec"
):
    CLASSES = GEN1_CLASSES if dataset == "gen1" else DSEC_CLASSES
    COLORS = GEN1_COLORS if dataset == "gen1" else DSEC_COLORS
    
    for i in range(len(x)):
        if scores is not None and scores[i] < conf:
            continue

        x0 = int(scale * x[i])
        y0 = int(scale * y[i])
        x1 = int(scale * (x[i] + w[i]))
        y1 = int(scale * (y[i] + h[i]))
        cls_id = int(labels[i])
        if cls_id < 0 or cls_id >= len(CLASSES):
            cls_id = 0

        color = (COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = f"{label}-{CLASSES[cls_id]}"
        if scores is not None and show_conf:
            text += f":{scores[i] * 100:.1f}"

        txt_color = (0, 0, 0) if np.mean(COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]

        cv2.rectangle(img, (x0, y0), (x1, y1), color, linewidth)
        txt_bk_color = (COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        txt_height = int(1.5 * txt_size[1])
        cv2.rectangle(
            img,
            (x0, max(0, y0 - txt_height)),
            (x0 + txt_size[0] + 1, y0 + 1),
            txt_bk_color,
            -1,
        )
        cv2.putText(
            img,
            text,
            (x0, max(0, y0 + txt_size[1] - txt_height)),
            font,
            0.4,
            txt_color,
            thickness=1,
        )
    return img


def render_object_detections_on_image(img, tracks, **kwargs):
    return _draw_bbox_on_img(
        img,
        tracks["x"],
        tracks["y"],
        tracks["w"],
        tracks["h"],
        tracks["class_id"],
        **kwargs,
    )

def _to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _xywh_center_to_topleft(xywh):
    x = xywh[:, 0] - xywh[:, 2] / 2.0
    y = xywh[:, 1] - xywh[:, 3] / 2.0
    w = xywh[:, 2]
    h = xywh[:, 3]
    return x, y, w, h


def _xywh_center_to_xyxy_torch(boxes):
    x1 = boxes[:, 0] - boxes[:, 2] / 2.0
    y1 = boxes[:, 1] - boxes[:, 3] / 2.0
    x2 = boxes[:, 0] + boxes[:, 2] / 2.0
    y2 = boxes[:, 1] + boxes[:, 3] / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _nms_torch(boxes_xyxy, scores, iou_thr=0.6):
    if boxes_xyxy.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes_xyxy.device)

    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]

        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        inter_w = (xx2 - xx1).clamp(min=0)
        inter_h = (yy2 - yy1).clamp(min=0)
        inter = inter_w * inter_h
        union = areas[i] + areas[rest] - inter + 1e-9
        iou = inter / union
        order = rest[iou <= iou_thr]

    return torch.tensor(keep, dtype=torch.long, device=boxes_xyxy.device)


def _decode_predictions(pred, conf_thr=0.25, nms_thr=0.6):
    # 1. REMOVE BATCH DIMENSION HERE
    # [1, N_anchors, 13] -> [N_anchors, 13]
    if pred.dim() == 3:
        pred = pred.squeeze(0)

    if pred.numel() == 0:
        return {
            "x": np.array([]),
            "y": np.array([]),
            "w": np.array([]),
            "h": np.array([]),
            "class_id": np.array([]),
            "score": np.array([]),
        }

    boxes_xywh = pred[:, :4]
    obj_scores = pred[:, 4]
    cls_scores, cls_ids = torch.max(pred[:, 5:], dim=1)
    scores = obj_scores * cls_scores

    keep = scores >= conf_thr
    if keep.sum() == 0:
        return {
            "x": np.array([]),
            "y": np.array([]),
            "w": np.array([]),
            "h": np.array([]),
            "class_id": np.array([]),
            "score": np.array([]),
        }

    boxes_xywh = boxes_xywh[keep]
    cls_ids = cls_ids[keep]
    scores = scores[keep]

    boxes_xyxy = _xywh_center_to_xyxy_torch(boxes_xywh)
    keep_idx = _nms_torch(boxes_xyxy, scores, iou_thr=nms_thr)

    boxes_xywh = _to_numpy(boxes_xywh[keep_idx])
    cls_ids = _to_numpy(cls_ids[keep_idx]).astype(np.int32)
    scores = _to_numpy(scores[keep_idx])
    x, y, w, h = _xywh_center_to_topleft(boxes_xywh)

    return {"x": x, "y": y, "w": w, "h": h, "class_id": cls_ids, "score": scores}


def _tracks_to_dict_tl_to_center(tracks_tensor, num_actual):
    if tracks_tensor.dim() == 3:
        tracks_tensor = tracks_tensor.squeeze(0)
    
    tracks = _to_numpy(tracks_tensor)[:num_actual]
    if tracks.shape[0] == 0:
        return {
            "x": np.array([]),
            "y": np.array([]),
            "w": np.array([]),
            "h": np.array([]),
            "class_id": np.array([]),
        }

    # Format is [class_id, cx, cy, w, h]
    # Index 0: class_id
    # Index 1: cx
    # Index 2: cy
    # Index 3: w
    # Index 4: h

    w = tracks[:, 3]
    h = tracks[:, 4]
    
    # Convert center coordinates back to top-left (x_tl, y_tl)
    x = tracks[:, 1] - (w / 2.0)
    y = tracks[:, 2] - (h / 2.0)
    
    cls = tracks[:, 0].astype(np.int32)
    
    return {"x": x, "y": y, "w": w, "h": h, "class_id": cls}


def _tracks_to_dict(tracks_tensor, num_actual):
    if tracks_tensor.dim() == 3:
        tracks_tensor = tracks_tensor.squeeze(0)
    
    tracks = _to_numpy(tracks_tensor)[:num_actual]
    if tracks.shape[0] == 0:
        return {
            "x": np.array([]),
            "y": np.array([]),
            "w": np.array([]),
            "h": np.array([]),
            "class_id": np.array([]),
        }

    # Format is [class_id, x_tl, y_tl, w, h]
    # Index 0: class_id
    # Index 1: x_tl
    # Index 2: y_tl
    # Index 3: w
    # Index 4: h

    w = tracks[:, 3]
    h = tracks[:, 4]
    x = tracks[:, 1]
    y = tracks[:, 2] 
    cls = tracks[:, 0].astype(np.int32)
    
    return {"x": x, "y": y, "w": w, "h": h, "class_id": cls}


def _find_latest_det_checkpoint():
    candidates = glob.glob("checkpoints/det_dvae_*/det_epoch_*.pth")
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def _call_detection_dataset(provider, detection_cfg, fallback_cfg):
    method_names = [
        "get_detection_test_dataset",
        "get_detection_val_dataset",
        "get_detection_train_dataset",
    ]
    method = None
    for name in method_names:
        if hasattr(provider, name):
            method = getattr(provider, name)
            break
    if method is None:
        raise AttributeError("DatasetProvider has no detection dataset getter.")

    signature = inspect.signature(method)
    available = set(signature.parameters.keys())
    kwargs_candidates = {
        "num_events": detection_cfg.get(
            "max_num_events", fallback_cfg.get("loader", {}).get("num_events", 100000)
        ),
        "max_num_events": detection_cfg.get(
            "max_num_events", fallback_cfg.get("loader", {}).get("num_events", 100000)
        ),
        "delta_t_ms": detection_cfg.get("delta_t_ms", 100),
        "num_classes": detection_cfg.get("num_classes", 8),
    }
    kwargs = {k: v for k, v in kwargs_candidates.items() if k in available}
    try:
        return method(**kwargs)
    except TypeError:
        return method()
