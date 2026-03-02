import os
import torch

# 1. SET THE ENVIRONMENT VARIABLE
# Your code explicitly looks for os.environ["GEN1_DATA_DIR"].
# Change this path to wherever your 'gen1' folder is located!
os.environ["GEN1_DATA_DIR"] = "/users/rpellerito/scratch/datasets"

from .sequence import Gen1 

def test_gen1_dataloader():
    print(f"Data Directory set to: {os.environ['GEN1_DATA_DIR']}")
    
    # 2. INSTANTIATE THE DATAMODULE
    # Tip: Set num_workers=0 for debugging. It forces the dataloader to run 
    # on the main thread, making error tracebacks much easier to read!
    print("Initializing Gen1 DataModule...")
    data_module = Gen1(
        batch_size=4, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=False
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
    batch = next(iter(train_loader))
    
    # 5. PRINT THE TENSOR SHAPES TO VERIFY
    print("\n✅ SUCCESS! Batch loaded successfully. Here are the shapes:")
    print("-" * 50)
    print(f"Batch index map (batch.batch): {batch.batch.shape}")
    print(f"Polarity / Features (batch.x): {batch.x.shape} -> Expected [Total_Events, 1]")
    print(f"Coordinates/Time (batch.pos):  {batch.pos.shape} -> Expected [Total_Events, 3]")
    
    if hasattr(batch, 'bbox'):
        print(f"Bounding Boxes (batch.bbox):   {batch.bbox.shape}")
        print(f"Box Labels (batch.y):          {batch.y.shape}")
        print(f"Batch Bbox map (batch_bbox):   {batch.batch_bbox.shape}")
        
    if hasattr(batch, 'edge_index'):
        print(f"Graph Edges (batch.edge_index):{batch.edge_index.shape}")
        print(f"Edge Attrs (batch.edge_attr):  {batch.edge_attr.shape}")
    print("-" * 50)
    
    # Optional: Look at the actual data of the first few events
    print("\nFirst 5 events (x, y, t) in the batch:")
    print(batch.pos[:5])
    print("First 5 polarities:")
    print(batch.x[:5])

if __name__ == "__main__":
    test_gen1_dataloader()