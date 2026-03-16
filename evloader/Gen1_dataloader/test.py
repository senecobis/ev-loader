import os
import torch
from evlicious import Events
import matplotlib.pyplot as plt
import numpy as np

os.environ["GEN1_DATA_DIR"] = "/users/rpellerito/scratch/datasets"

# from .sequence import Gen1
from .PatchedSequence import PatchedGen1 
from ..utils.utils_detection import render_object_detections_on_image

DEBUG = True

def visualise_patch(x, y, p, t, width, height, sequence_id):
    t = t.astype(np.int64)
    p = p.astype(np.int8)
    ev = Events(x=x, y=y, p=p, t=t, width=width, height=height)
    rendered = ev.render()
    plt.imshow(rendered)
    plt.title(f"Visualisation of patch events (N={len(ev)})")
    os.makedirs("debug_viz", exist_ok=True)

    plt.savefig(f"debug_viz/patch_viz_{sequence_id}.png")
    
def visualise_stitched_patches(batch, batch_idx, n_h, n_w, img_w, img_h, sequence_id):
    """
    Stitches local-coordinate patches from a dense batch back into a global image.
    """
    # 1. Extract data for the specific batch index
    # x is (Max_S, 4) -> [x_local, y_local, t, p]
    events = batch.x[batch_idx]
    # cu_seqlens is (P + 1)
    cu_seqlens = batch.cu_seqlens[batch_idx]
    
    # Calculate patch dimensions
    patch_w = img_w // n_w
    patch_h = img_h // n_h
    
    stitched_image = np.zeros((img_h, img_w), dtype=np.uint8)
    
    # 2. Iterate through each patch ID (P = n_h * n_w)
    num_patches = n_h * n_w
    for patch_id in range(num_patches):
        # Get start and end indices for this patch from cu_seqlens
        start_idx = cu_seqlens[patch_id]
        end_idx = cu_seqlens[patch_id + 1]
        
        if start_idx == end_idx:
            continue  # No events in this patch
            
        # Extract events for this specific patch
        patch_events = events[start_idx:end_idx]
        
        # Local coordinates (assuming x is index 0, y is index 1)
        x_local = patch_events[:, 0].numpy().astype(np.int32)
        y_local = patch_events[:, 1].numpy().astype(np.int32)
        
        # 3. Calculate Global Offsets
        col = patch_id % n_w
        row = patch_id // n_w
        x_offset = col * patch_w
        y_offset = row * patch_h
        
        x_global = x_local + x_offset
        y_global = y_local + y_offset
        
        # Clip coordinates to stay within image bounds (safety check)
        mask = (x_global < img_w) & (y_global < img_h)
        
        # 4. Draw onto the image
        stitched_image[y_global[mask], x_global[mask]] = 255
    
    plt.figure(figsize=(10, 8))
    plt.imshow(stitched_image, cmap='gray')
    plt.title(f"Stitched Patches (Local -> Global) | Seq: {sequence_id}")
    os.makedirs("debug_viz", exist_ok=True)
    plt.savefig(f"debug_viz/stitched_{sequence_id}.png")
    plt.close()
    return stitched_image

def test_gen1_dataloader():
    print(f"Data Directory set to: {os.environ['GEN1_DATA_DIR']}")
    
    # 2. INSTANTIATE THE DATAMODULE
    # Tip: Set num_workers=0 for debugging. It forces the dataloader to run 
    # on the main thread, making error tracebacks much easier to read!
    print("Initializing Gen1 DataModule...")
    if DEBUG:
        print("⚠️ DEBUG MODE: Setting num_workers=0 for easier tracebacks.")
        num_workers = 0
    else:
        num_workers = 64
    preprocess_again = False
    
    data_module = PatchedGen1(
        batch_size=30, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=False,
        num_events_per_sample=100000,
        n_patches_h=60,
        n_patches_w=72,
        preprocess_again=preprocess_again
    )
    
    # 3. RUN THE LIGHTNING LIFECYCLE
    print("Running prepare_data() ... (This will process .dat to .pkl if needed)")
    if preprocess_again:
        data_module.prepare_data()
    
    print("Running setup() ... (This builds the train/val dataset objects)")
    data_module.setup()
    
    # Check if the datasets actually populated
    print(f"Training samples: {len(data_module.train_dataset)}")
    print(f"Validation samples: {len(data_module.val_dataset)}")
    
    if len(data_module.train_dataset) == 0:
        print("🚨 ERROR: No training data found! Check your GEN1_DATA_DIR path.")
        return

    # 4. FETCH A BATCH AND INSPECT IT
    print("\nFetching a batch from the train_dataloader...")
    train_loader = data_module.train_dataloader()
    
    # Grab the very first batch
    for batch in train_loader:
        print("-" * 50)
        batched = batch.x
        print(f"Patched Event Tensor (batch.x_patched): {batched.shape}")
        patch_ids = batch.patch_ids
        print(f"Patch IDs (batch.patch_ids): {patch_ids.shape}")
        cu_seqlens = batch.cu_seqlens
        print(f"Cumulative Sequence Lengths (batch.cu_seqlens): {cu_seqlens.shape}")
        mask = batch.mask
        print(f"Padding Mask (batch.mask): {mask.shape}")
        padded_bbox = batch.bbox_padded
        print(f"Padded Bounding Boxes (batch.bbox_padded): {padded_bbox.shape}")
        print(f"Original Bounding Boxes (batch.bbox): {batch.bbox.shape}")
        print(f"Batch BBox Map (batch.batch_bbox): {batch.batch_bbox.shape}")
        
        stitched_image = visualise_stitched_patches(
            batch=batch,
            batch_idx=0,
            n_h=data_module.n_h,
            n_w=data_module.n_w,
            img_w=data_module.dims[0], # 304
            img_h=data_module.dims[1], # 240
            sequence_id="batch_0_stitched"
        )
        bbox_batch_mask = batch.batch_bbox==0
        tracks = {
            "class_id": batch.bbox[:, 0].numpy().astype(np.int32),  # class_id
            "x": batch.bbox[bbox_batch_mask, 1].numpy(),  # x
            "y": batch.bbox[bbox_batch_mask, 2].numpy(),  # y
            "w": batch.bbox[bbox_batch_mask, 3].numpy(),  # w
            "h": batch.bbox[bbox_batch_mask, 4].numpy()  # h
        }
        gt_img = render_object_detections_on_image(
            stitched_image.copy(), tracks, label="gt", linewidth=2, show_conf=False
            )
        plt.figure(figsize=(10, 8))
        plt.imshow(gt_img)
        plt.title(f"Stitched Patches with GT BBoxes | Seq: batch_0")
        plt.savefig(f"debug_viz/bbx_of_batch_0_stitched.png")
        plt.close()

        if hasattr(batch, 'bbox'):
            print(f"Bounding Boxes (batch.bbox):   {batch.bbox.shape}")
            print(f"Box Labels (batch.y):          {batch.y.shape}")
            print(f"Batch Bbox map (batch_bbox):   {batch.batch_bbox.shape}")
            
        # Optional: Look at the actual data of the first few events
        print("\nFirst 5 events (x, y, p, t) in the batch:")
        print(batch.x[:5])
        print(f"Total events in batch: {len(batch.x)}")
        print("-" * 50)

if __name__ == "__main__":
    test_gen1_dataloader()