"""
Script to load EVIMO v1 dataset for training by Roberto Pellerito rpellerito@ifi.uzh.ch
"""
import re
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from .EVIMOTestSequence import EVIMOTestSequence
from ..representations.representations import EventToStack_Numpy
from ev_loader.representations.representations import VoxelGrid
from ev_loader.DSEC_dataloader.Sequence import Sequence


class UnpackedEVIMO():
    def __init__(self, scene_path: str):
        self.scene_path = scene_path
        self.images_folder_path = Path(f"{scene_path}/images")
        self.poses_path = Path(f"{scene_path}/poses.npy")
        self.timestamps_path = Path(f"{scene_path}/timestamps.npy")

        files = [p.name for p in self.images_folder_path.iterdir() if p.is_file()]
        self.images_paths = sorted(files, key=self.natural_key)
        self.poses = torch.from_numpy(np.load(self.poses_path))
        self.timestamps = torch.from_numpy(np.load(self.timestamps_path))

    @staticmethod
    def natural_key(s):
        # split into [text, number, text, number…]
        parts = re.split(r'(\d+)', s)
        # convert digit chunks to int
        return [int(p) if p.isdigit() else p.lower() for p in parts]
    
    def __len__(self):
        # Return the number of images
        return len(self.images_paths)
    
    def get_image(self, idx):
        image_path = self.images_folder_path / self.images_paths[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        return image

    def __getitem__(self, idx):
        pose = self.poses[idx]
        image = self.get_image(idx)
        return image, pose
    

class EVIMOFramesSequence(EVIMOTestSequence):
    def __init__(self, h5_path: str, window_ms, num_bins, representation="voxel"):
        super().__init__(h5_path, window_ms, num_bins, representation)
        id_ = self.h5_path.split("/")[-1].split(".")[0]
        scene = os.path.dirname(h5_path) + f"/{id_}"
        self.unpacked_evimo = UnpackedEVIMO(scene)
        self.stack_rapresentation = EventToStack_Numpy(num_bins=num_bins, height=self.height, width=self.width)
        
        # TODO Hardcoded num bins =2 to use them for dynamic masker / Herm
        self.voxel_grid = VoxelGrid(channels=2, height=self.height, width=self.width, normalize=True)
        
    def _events_to_voxel_grid(self, x, y, p, t):
        return Sequence.events_to_voxel_grid(self.voxel_grid, x, y, p, t)

    def get_representation(self, x, y, t, p, representation):
        if representation == 'stack':
            return torch.from_numpy(self.stack_rapresentation(x=x, y=y, p=p))
        elif representation == 'voxel':
            voxel = self._events_to_voxel_grid(x=x, y=y, p=p, t=t)
            return voxel
        elif representation == 'oms':
            p[p==0] = 1.0
            return self._events_to_voxel_grid(x=x, y=y, p=p, t=t)
        else:
            raise ValueError(f"Representation {representation} not supported.")
    
    def get_single_item(self, idx):
        with torch.no_grad():
            id0 = self.start_ev_ind[idx]
            id1 = self.end_ev_ind[idx]

            events = self.events[id0:id1]
            ev_x = events[:, 0]
            ev_y = events[:, 1]
            ev_t = events[:, 2]
            ev_p = events[:, 3]

            dt = ev_t[-1] - ev_t[0]
            voxel = self.get_representation(x=ev_x, y=ev_y, p=ev_p, t=ev_t, representation='voxel')
            stack = self.get_representation(x=ev_x, y=ev_y, p=ev_p, t=ev_t, representation='stack')
            binary_mask = self.convert_to_binary_mask(self.mask[idx])

            image = self.unpacked_evimo.get_image(idx)
            pose = self.unpacked_evimo.poses[idx]
            intrinsics = torch.as_tensor([self.fx[()], self.fy[()], self.cx[()], self.cy[()]])

            return {
                "voxel": voxel,
                "stack": stack,
                "image": torch.from_numpy(image),
                "pose": pose,
                "intrinsics": intrinsics,
                "mask": torch.from_numpy(self.mask[idx]),
                "dynamic_mask": torch.from_numpy(binary_mask).unsqueeze(0).float(),
                "sampled_dt": torch.tensor([dt], dtype=torch.float32, requires_grad=False),
                "sequence_id": self.sequence_id
            }
    
    def __getitem__(self, idx):        
        return self.get_single_item(idx)
    

if __name__ == "__main__":
    # Example usage
    # h5_path = "/home/rpg/Downloads/EVIMO1/test/box/seq_00.h5"
    h5_path = "/data/scratch/pellerito/datasets/EVIMO1/train/box/seq_00.h5"

    dataset = EVIMOFramesSequence(h5_path, window_ms=50, num_bins=2, representation="oms")
    print(f"Number of samples: {len(dataset)}")
    
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"Sample {i}: {sample}")