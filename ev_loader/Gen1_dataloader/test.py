import os
import torch
from evlicious import Events
import matplotlib.pyplot as plt
import numpy as np

os.environ["GEN1_DATA_DIR"] = "/users/rpellerito/scratch/datasets"

# from .sequence import Gen1
from .PatchedSequence import PatchedGen1 

def visualise_patch(x, y, p, t, width, height, sequence_id):
    t = t.astype(np.int64)
    p = p.astype(np.int8)
    ev = Events(x=x, y=y, p=p, t=t, width=width, height=height)
    rendered = ev.render()
    plt.imshow(rendered)
    plt.title(f"Visualisation of patch events (N={len(ev)})")
    os.makedirs("debug_viz", exist_ok=True)

    plt.savefig(f"debug_viz/patch_viz_{sequence_id}.png")

def test_gen1_dataloader():
    print(f"Data Directory set to: {os.environ['GEN1_DATA_DIR']}")
    
    # 2. INSTANTIATE THE DATAMODULE
    # Tip: Set num_workers=0 for debugging. It forces the dataloader to run 
    # on the main thread, making error tracebacks much easier to read!
    print("Initializing Gen1 DataModule...")
    data_module = PatchedGen1(
        batch_size=16, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=False,
        num_events_per_sample=100000,
        n_patches_h=16,
        n_patches_w=16,
        preprocess_again=True
    )
    
    # 3. RUN THE LIGHTNING LIFECYCLE
    print("Running prepare_data() ... (This will process .dat to .pkl if needed)")
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
    
        # 5. PRINT THE TENSOR SHAPES TO VERIFY
        print("\n✅ SUCCESS! Batch loaded successfully. Here are the shapes:")
        print("-" * 50)
        print(f"Batch index map (batch.batch): {batch.batch.shape}")
        print(f"Polarity / Features (batch.x): {batch.x.shape} -> Expected [Total_Events, 4]")
        
        if hasattr(batch, 'bbox'):
            print(f"Bounding Boxes (batch.bbox):   {batch.bbox.shape}")
            print(f"Box Labels (batch.y):          {batch.y.shape}")
            print(f"Batch Bbox map (batch_bbox):   {batch.batch_bbox.shape}")
            
        print("-" * 50)
        
        # Optional: Look at the actual data of the first few events
        print("\nFirst 5 events (x, y, p, t) in the batch:")
        print(batch.x)

if __name__ == "__main__":
    test_gen1_dataloader()