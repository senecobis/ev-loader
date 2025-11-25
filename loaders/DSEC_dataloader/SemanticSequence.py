import numpy as np
import cv2
import torch
import json
from pathlib import Path
from .sequence import Sequence
from ..utils.labels import labels_11, labels_19
import pycocotools.mask as mask_util
from ..utils.img_event_ref_transform_DSEC import TransformImageToEventRef
from matplotlib import pyplot as plt


class SemanticSequence(Sequence):
    """
    Extends the Sequence class to additionally load semantic labels for DSEC dataset.
    Supports both 11-class and 19-class formats and converts to COCO format for evaluation.
    """

    def __init__(self, *args, class_format='19', **kwargs):
        """
        Args:
            class_format (str): Which class format to use ('11' or '19')
            *args, **kwargs: Arguments for the base Sequence class.
        """
        super().__init__(*args, **kwargs)
        self.labels = []
        
        self.class_format = class_format
        assert self.class_format in ['11', '19'], "class_format must be '11' or '19'"
        
        semantic_11_classes_dir = Path(self.sequence_path) / '11classes'
        semantic_19_classes_dir = Path(self.sequence_path) / '19classes'
        
        self.semantics_exists = False
        
        # Check which formats are available
        self.has_11classes = semantic_11_classes_dir.exists()
        self.has_19classes = semantic_19_classes_dir.exists()
        
        if self.has_11classes and self.class_format == '11':
            self.semantics_exists = True
            self.semantic_label_dir = str(semantic_11_classes_dir)
            self.semantic_label_paths = sorted([str(p) for p in semantic_11_classes_dir.glob('*.png')])
            self.labels = labels_11
        elif self.has_19classes and self.class_format == '19':
            self.semantics_exists = True
            self.semantic_label_dir = str(semantic_19_classes_dir)
            self.semantic_label_paths = sorted([str(p) for p in semantic_19_classes_dir.glob('*.png')])
            self.labels = labels_19
        elif self.has_11classes and not self.has_19classes and self.class_format == '19':
            # Fallback to 11 classes if 19 not available
            print(f"Warning: 19-class format requested but not available for {self.sequence_path}. Using 11-class format.")
            self.semantics_exists = True
            self.semantic_label_dir = str(semantic_11_classes_dir)
            self.semantic_label_paths = sorted([str(p) for p in semantic_11_classes_dir.glob('*.png')])
            self.labels = labels_11
            self.class_format = '11'
        elif self.has_19classes and not self.has_11classes and self.class_format == '11':
            # Fallback to 19 classes if 11 not available
            print(f"Warning: 11-class format requested but not available for {self.sequence_path}. Using 19-class format.")
            self.semantics_exists = True
            self.semantic_label_dir = str(semantic_19_classes_dir)
            self.semantic_label_paths = sorted([str(p) for p in semantic_19_classes_dir.glob('*.png')])
            self.labels = labels_19
            self.class_format = '19'
        
        # Remove first semantics path if it exists (to match the base class behavior)
        if self.semantics_exists and len(self.semantic_label_paths) > 0:
            first_path = Path(self.semantic_label_paths[0])
            if int(first_path.stem) == 0:
                self.semantic_label_paths.pop(0)
        
        # Create COCO category mapping if semantics available
        if self.semantics_exists:
            self.create_coco_categories()

        self.img_transf = TransformImageToEventRef(conf=self.intrinsics, height=1080, width=1440)

        self.frames_pathstrings_original = self.frames_pathstrings_original[1:]  # Skip first frame as per base class behavior
        timestamps = np.array(self.timestamps)
        duplicated = np.repeat(timestamps, 2) # duplicate timestamps to align with frames
        self.timestamps = duplicated


    def create_coco_categories(self):
        """
        Create COCO categories based on the selected class format.
        """
        self.categories = []
        
        for label in self.labels:
            # Skip labels that are ignored in evaluation
            if label.ignoreInEval:
                continue
                
            # Create COCO category entry
            category = {
                'id': int(label.trainId) if label.trainId != 255 else 0,  # Use trainId as COCO category id
                'name': label.name,
                'supercategory': label.category
            }
            self.categories.append(category)

    def load_semantic_image(self, filepath):
        """
        Load a semantic segmentation image.
        
        Args:
            filepath (str): Path to the semantic segmentation image.
            
        Returns:
            numpy.ndarray: Semantic segmentation image with class IDs.
        """
        # Load raw semantic segmentation image (contains class IDs)
        semantic_img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        
        if semantic_img is None:
            raise ValueError(f"Could not load semantic image: {filepath}")
            
        return semantic_img

    def create_coco_annotations(self, semantic_img, image_id):
        """
        Create COCO annotations from a semantic segmentation image.
        
        Args:
            semantic_img (numpy.ndarray): Semantic segmentation image with class IDs.
            image_id (int): Image ID for COCO annotations.
            
        Returns:
            list: List of COCO annotation dictionaries.
        """
        if mask_util is None:
            return []
            
        annotations = []
        
        # Find unique class IDs in the image (exclude ignore regions with trainId=255)
        unique_ids = np.unique(semantic_img)
        unique_ids = unique_ids[unique_ids != 255]
        
        annotation_id = 1
        
        for class_id in unique_ids:
            # Create binary mask for this class
            binary_mask = (semantic_img == class_id).astype(np.uint8)
            
            # Skip if no pixels for this class
            if binary_mask.sum() == 0:
                continue
                
            # Find contours for creating polygons
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Skip small regions
            if len(contours) == 0:
                continue
                
            # Create RLE encoding for the mask
            rle = mask_util.encode(np.asfortranarray(binary_mask))
            rle['counts'] = rle['counts'].decode('utf-8')  # Convert bytes to string for JSON
            
            # Find bounding box
            x, y, w, h = cv2.boundingRect(binary_mask)
            
            # Create annotation
            annotation = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': int(class_id),
                'segmentation': rle,
                'area': float(binary_mask.sum()),
                'bbox': [x, y, w, h],
                'iscrowd': 0
            }
            
            annotations.append(annotation)
            annotation_id += 1
            
        return annotations

    def frame_gt(self, index):
        frame_path = self.frames_pathstrings_original[index]
        frame = self.get_frame(frame_path)
        return frame
    
    @staticmethod
    def get_transformed_frame(transform, frame):
        transformed_frame = transform(frame)
        # Crop to 640x440 based on https://github.com/uzh-rpg/DSEC/issues/25
        # It must have the same shape of the segmentation
        transformed_frame = transformed_frame[:440, :640, :]
        # Convert to tensor and change channel order
        return torch.from_numpy(transformed_frame).int()
    
    def __getitem__(self, idx):
        # Get the base data from Sequence
        data = super().__getitem__(idx)
        data['frame'] = self.get_transformed_frame(self.img_transf, data['frame'])
        
        # Add semantic segmentation if available
        if self.semantics_exists and idx < len(self.semantic_label_paths):
            semantic_path = self.semantic_label_paths[idx]
            semantic_img = self.load_semantic_image(semantic_path)

            # Convert to tensor
            semantic_tensor = torch.from_numpy(semantic_img).long()
            
            # Add to data dictionary
            data['semantic_gt'] = semantic_tensor
            data['semantic_format'] = self.class_format
            
            # For COCO format, create annotations for this specific image
            image_id = data['file_index']
            coco_annotations = self.create_coco_annotations(semantic_img, image_id)
            data['coco_annotations'] = coco_annotations

            # return events just for plotting
            data['events'] = {
                'x': self.x_rect,
                'y': self.y_rect,
                'p': self.p,
                't': self.t
            }

    
        return data
    
if __name__ == "__main__":
    from evlicious import Events

    # Use the specific sequence path provided by the user
    sequence_path = Path("/data/scratch/pellerito/datasets/DSEC/test/zurich_city_14_c")

    seq = SemanticSequence(sequence_path, class_format='19')
    # data = seq[0]
    for data in seq:
        # print(f"Data: {data}")

        raw_events = data['events']
        mask_for_out_events = (raw_events['x'] >= 0) & (raw_events['x'] < 640) & (raw_events['y'] >= 0) & (raw_events['y'] < 440)
        events = Events(x=raw_events['x'][mask_for_out_events].astype(np.int16).astype(np.uint16), 
                        y=raw_events['y'][mask_for_out_events].astype(np.int16).astype(np.uint16), 
                        t=raw_events['t'][mask_for_out_events].astype(np.int64), 
                        p=raw_events['p'][mask_for_out_events].astype(np.int8), 
                        width=640, height=440
                        )
        frame = np.array(data['frame'])
        rendered_events = events.render(frame)
        plt.imshow(rendered_events)
        plt.savefig('rendered_events.png')

