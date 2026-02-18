import numpy as np
from functools import lru_cache


class BaseDirectory:
    def __init__(self, root):
        self.root = root


class TracksDirectory(BaseDirectory):
    @property
    @lru_cache(maxsize=1)
    def tracks(self):
        return np.load(self.root / "object_detections/left/tracks.npy")


def compute_img_idx_to_track_idx(t_track, t_image):
    x, counts = np.unique(t_track, return_counts=True)
    i, j = (x.reshape((-1, 1)) == t_image.reshape((1, -1))).nonzero()
    deltas = np.zeros_like(t_image)
    deltas[j] = counts[i]
    idx = np.concatenate([np.array([0]), deltas]).cumsum()
    return np.stack([idx[:-1], idx[1:]], axis=-1).astype("uint64")


class TracksLoader:
    """
    Single-sequence tracks-only loader.
    Mirrors DSECDet index-to-track behavior without loading events/images.
    
    Each sample of this dataset loads one image, events, and labels at a timestamp. The behavior is different for 
    sync='front' and sync='back', and these are visualized below.

    Legend: 
    . = events
    | = image
    L = label

    sync='front'
    -------> time
    .......|
            L

    sync='back'
    -------> time
    |.......
            L
    """

    def __init__(self, sequence_root, sync="front", timestamps_images=None):
        assert sequence_root.exists()
        assert sync in ["front", "back"]

        self.sync = sync
        self.directory = TracksDirectory(sequence_root)
        if timestamps_images is not None:
            self.image_timestamps = timestamps_images
        else:
            self.image_timestamps = np.genfromtxt(sequence_root / "images/timestamps.txt", dtype="int64")
        self.img_idx_to_track_idx = compute_img_idx_to_track_idx(
            self.directory.tracks["t"], self.image_timestamps
        )

    def __len__(self):
        return len(self.img_idx_to_track_idx) - 1

    def __getitem__(self, index):
        return self.get_tracks(index)

    @staticmethod
    def get_index_window(index, num_idx, sync="back"):
        if sync == "front":
            assert 0 < index < num_idx
            i_0 = index - 1
            i_1 = index
        else:
            i_0 = index
            i_1 = np.clip(index + 1, 0, num_idx - 1)
        return i_0, i_1

    def index_to_track_index(self, index):
        i_0, _ = self.get_index_window(index, len(self.img_idx_to_track_idx), sync=self.sync)
        idx0, idx1 = self.img_idx_to_track_idx[i_0]
        return int(idx0), int(idx1)

    def get_tracks(self, index, mask=None):
        idx0, idx1 = self.index_to_track_index(index)
        tracks = self.directory.tracks[idx0:idx1]
        if mask is not None:
            tracks = tracks[mask[idx0:idx1]]
        return tracks

    def get_relative_tracks(self, index, mask=None):
        return self.get_tracks(index=index, mask=mask)
