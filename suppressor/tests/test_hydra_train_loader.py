from pathlib import Path
import random
import time
import numpy as np
import torch
from ..DSEC_dataloader.provider import DatasetProvider
from ..utils.utils_collate import custom_collate

def seed_worker(worker_id):
    """Ensure each worker has a deterministic seed"""
    seed = torch.initial_seed() % (2**32)  # Get unique seed for each worker
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    # dsec_path = Path("/data/scratch/pellerito/datasets/DSEC")
    dsec_path = Path("/home/rpg/Downloads/DSEC")

    dataset_provider = DatasetProvider(
        dsec_path, num_bins=2, representation='voxel', delta_t_ms=50)
    
    train_dataset = dataset_provider.get_hydra_train_dataset(
        sequence_len=5, # passes_loss corresponds to sequence len
        max_num_grad_events=10000,
        dt = [100, 100],
        augment=["Horizontal", "Vertical", "Polarity"],
        augment_prob=[0.5, 0.5, 0.5],
        )
    
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            drop_last=True,
            batch_size=1,
            shuffle=True,
            num_workers=1,
            worker_init_fn=seed_worker,
            pin_memory=True,
            # collate_fn=custom_collate
            )
    start = time.time()
    for batch_idx, batch_data in enumerate(train_loader):
        #running average time
        running_avg_time = (time.time() - start) / (batch_idx + 1)
        print(f"Loading batch {batch_idx} took as running avg {running_avg_time} seconds")
        start = time.time()
        for data_ind in range(len(batch_data)-1):
            data_t0 = batch_data[data_ind]
            event_voxel = data_t0["representation"]
