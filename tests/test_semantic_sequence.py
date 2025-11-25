from suppressor.DSEC_dataloader.SemanticSequence import SemanticSequence
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Use the specific sequence path provided by the user
sequence_path = Path("/data/scratch/pellerito/datasets/DSEC/test/zurich_city_14_c")

seq = SemanticSequence(sequence_path, class_format='19')
data = seq[0]

# Find category IDs for car, bike, pedestrian
target_names = {"car", "bicycle", "person", "pedestrian"}
cat_ids = [cat["id"] for cat in seq.categories if cat["name"].lower() in target_names]
print("Evaluating for category IDs (car, bike, pedestrian):", cat_ids)

# Build a minimal COCO dataset for evaluation
image_id = data["file_index"]
coco_gt_dict = {
    "images": [
        {
            "id": image_id,
            "width": data["semantic_gt"].shape[1],
            "height": data["semantic_gt"].shape[0],
            "file_name": f"{image_id}.png"
        }
    ],
    "annotations": data["coco_annotations"],
    "categories": seq.categories
}

# Prepare predictions: add 'score' field to each annotation
pred_annotations = []
for ann in data["coco_annotations"]:
    pred_ann = ann.copy()
    pred_ann["score"] = 1.0
    pred_annotations.append(pred_ann)


# --- Create in-memory COCO GT object ---
coco_gt = COCO()
coco_gt.dataset = coco_gt_dict
coco_gt.createIndex()

# --- Create in-memory COCO DT object from list of predictions ---
coco_dt = coco_gt.loadRes(pred_annotations)

# --- Run COCO evaluation ---
coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
coco_eval.params.imgIds = [image_id]
coco_eval.params.catIds = cat_ids
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()