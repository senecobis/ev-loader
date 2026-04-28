import numpy as np
import torch
import hdf5plugin
import h5py

from pathlib import Path
from typing import Optional, Callable
from torch.utils.data import Dataset
from torch_geometric.data import Data
from .augment import init_transforms
from .utils import to_data
from ..utils.VectorizedPatchfier import VectorizedPatchfier


class NCaltech101(Dataset):

    def __init__(
        self,
        root: Path,
        split,
        transform: Optional[Callable[[Data], Data]] = None,
        num_events: int = 50000,
        n_patches_h: Optional[int] = None,
        n_patches_w: Optional[int] = None,
        max_patch_events: Optional[int] = 100,
        patch_to_local_coords: bool = True,
    ):
        super().__init__()
        assert split in ["training", "testing", "validation"], \
                    "split must be either 'training', 'testing', or 'validation'"
        self.load_dir = root / split
        self.classes = sorted([d.name for d in self.load_dir.glob("*")])
        self.num_classes = len(self.classes)
        self.files = sorted(list(self.load_dir.rglob("*.h5")))
        self.height = 180
        self.width = 240
        if transform is not None and hasattr(transform, "transforms"):
            init_transforms(transform.transforms, self.height, self.width)
        self.transform = transform
        self.time_window = 1000000
        self.num_events = num_events
        self.n_h = n_patches_h
        self.n_w = n_patches_w
        self.max_patch_events = max_patch_events
        self.patch_to_local_coords = patch_to_local_coords
        self.patchfier = None
        if self.n_h is not None or self.n_w is not None:
            assert self.n_h is not None and self.n_w is not None, \
                "n_patches_h and n_patches_w must be provided together"
            self.patchfier = VectorizedPatchfier(
                input_H=self.height,
                input_W=self.width,
                patch_grid_H=self.n_h,
                patch_grid_W=self.n_w,
            )

    def __len__(self):
        return len(self.files)

    def preprocess(self, data):
        data.t -= (data.t[-1] - self.time_window + 1)
        return data

    def load_events(self, f_path):
        return _load_events(f_path, self.num_events)

    def __getitem__(self, idx):
        f_path = self.files[idx]
        target = self.classes.index(str(f_path.parent.name))

        events = self.load_events(f_path)
        data = to_data(**events,  bbox=self.load_bboxes(f_path, target),
                       t0=events['t'][0], t1=events['t'], width=self.width, height=self.height,
                       time_window=self.time_window)

        data = self.preprocess(data)

        data = self.transform(data) if self.transform is not None else data

        if not hasattr(data, "t"):
            data.t = data.pos[:, -1:]
            data.pos = data.pos[:, :2].type(torch.int16)

        if self.patchfier is not None:
            self._add_patchified_stream(data)

        return data

    def _add_patchified_stream(self, data: Data):
        events = torch.cat([
            data.pos.to(torch.float32),
            data.t.reshape(-1, 1).to(torch.float32),
            data.x.reshape(-1, 1).to(torch.float32),
        ], dim=1).unsqueeze(0)

        patched_events, mask = self.patchfier(
            events,
            max_seq_len=self.max_patch_events,
            to_local_coords=self.patch_to_local_coords,
        )

        _, num_patches, max_seq_len, num_channels = patched_events.shape
        x_patched = patched_events.reshape(-1, num_channels)
        mask_long = mask.reshape(-1)
        patch_ids = torch.arange(
            num_patches,
            dtype=torch.long,
            device=patched_events.device,
        ).repeat_interleave(max_seq_len)

        x_patched = x_patched[mask_long]
        patch_ids = patch_ids[mask_long]

        counts = torch.bincount(patch_ids, minlength=num_patches)
        cu_seqlens = torch.zeros(
            counts.shape[0] + 1,
            dtype=torch.int32,
            device=patched_events.device,
        )
        torch.cumsum(counts, dim=0, out=cu_seqlens[1:])

        data.x_patched = x_patched
        data.patchified_stream = x_patched
        data.patch_ids = patch_ids
        data.cu_seqlens = cu_seqlens
        data.patched_events = patched_events
        data.mask = mask
        data.n_patches_h = self.n_h
        data.n_patches_w = self.n_w
        data.num_patches = num_patches
        data.patch_h = self.patchfier.patch_size_h
        data.patch_w = self.patchfier.patch_size_w

    def load_bboxes(self, raw_file: Path, class_id):
        rel_path = str(raw_file.relative_to(self.load_dir))
        rel_path = rel_path.replace("image_", "annotation_").replace(".h5", ".bin")
        annotation_file = self.load_dir / "../annotations" / rel_path
        with annotation_file.open() as fh:
            annotations = np.fromfile(fh, dtype=np.int16)
            annotations = np.array(annotations[2:10])

        return np.array([
            annotations[0], annotations[1],  # upper-left corner
            annotations[2] - annotations[0],  # width
            annotations[5] - annotations[1],  # height
            class_id,
            1
        ]).astype("float32").reshape((1,-1))

def _load_events(f_path, num_events):
    with h5py.File(str(f_path)) as fh:
        fh = fh['events']
        x = fh["x"][-num_events:]
        y = fh["y"][-num_events:]
        t = fh["t"][-num_events:]
        p = fh["p"][-num_events:]
    return dict(x=x, y=y, t=t, p=p)

if __name__ == '__main__':
    dataset = NCaltech101(
        root=Path("/users/rpellerito/scratch/datasets/ncaltech101"), 
        split="training", 
        transform=None,
        n_patches_h=18,
        n_patches_w=24,
        max_patch_events=512,
        )
    data = dataset[0]
    print(data)
