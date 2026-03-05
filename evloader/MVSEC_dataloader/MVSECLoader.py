import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path


class MVSECLoader():
    def __init__(self, data_path, gt_path):
        """Load the DSEC dataset from the given paths
        Args:
            images_path (str): Path to the images folder
            events_path (str): Path to the events file
            tstamps_path (str): Path to the timestamps file
            K_path (str): Path to the intrinsics file
            disparity_path (str): Path to the disparity images
        """
        data_path = Path(data_path)
        gt_path = Path(gt_path)

        self.data_path = data_path
        self.gt_path = gt_path
        sensor_data = self.h5py_loader(data_path)
        events_all = sensor_data["ev"]
        ev_ts = sensor_data["ev_ts"]
        images = sensor_data["images"]
        images_ts = sensor_data["images_ts"]
        events_indices = sensor_data["events_indices"]

        GT_data = self.h5py_gt_loader(gt_path)
        poses = GT_data["poses"]
        poses_ts = GT_data["timestamps"]
        depths = GT_data["depth_rectified"]
        depths_ts = GT_data["depth_rectified_ts"]
        flow = GT_data["flow_dist"]
        flow_ts = GT_data["flow_dist_ts"]\
        # TODO: import calibarion also for indoor scenes
        K, (width, height) = self.load_intrinsics_outdoor()
        D = self.load_distortion_outdoor()

        # calibration
        self.calibration = K
        self.distortion = D
        self.image_shape = (width, height)
        # Events
        self.events = events_all
        self.events_ts = ev_ts
        self.events_indices = events_indices
        # Images
        self.images = images
        self.images_ts = images_ts
        # 6D poses
        self.T = poses
        self.poses_ts = poses_ts
        # Depths
        self.depths = depths
        self.depths_ts = depths_ts
        # Flow
        self.flow = flow
        self.flow_ts = flow_ts

    @staticmethod
    def h5py_loader(path: str):
        """Basic loader for .hdf5 files.
        Args:
            path (str) ... Path to the .hdf5 file.

        Returns:
            timestamp (dict) ... Doctionary of numpy arrays. Keys are "left" / "right".
            davis_left (dict) ... "event": np.ndarray.
            davis_right (dict) ... "event": np.ndarray.
        """
        data = h5py.File(path, "r")
        out = {}
        out["ev_ts"] = np.array(data["davis"]["left"]["events"][:, 2])
        out["ev"] = np.array(data["davis"]["left"]["events"], dtype=np.float64)
        out["images"] = np.array(data["davis"]["left"]["image_raw"])
        out["images_ts"] = np.array(data["davis"]["left"]["image_raw_ts"])
        out["events_indices"] = np.array(data["davis"]["left"]["image_raw_event_inds"])
        data.close()
        return out

    @staticmethod
    def h5py_gt_loader(path: Path):
        data = h5py.File(path, "r")
        poses = data["davis"]["left"]["pose"][:]
        timestamps = data["davis"]["left"]["pose_ts"][:]
        flow_dist = data["davis"]["left"]["flow_dist"][:]
        flow_dist_ts = data["davis"]["left"]["flow_dist_ts"][:]
        depth_rectified = data["davis"]["left"]["depth_image_rect"][:]
        depth_rectified_ts = data["davis"]["left"]["depth_image_rect_ts"][:]
        data.close()
        out = {
            "poses": poses,
            "timestamps": timestamps,
            "depth_rectified": depth_rectified,
            "depth_rectified_ts": depth_rectified_ts,
            "flow_dist": flow_dist,
            "flow_dist_ts": flow_dist_ts,
        }
        return out

    @staticmethod
    def load_intrinsics_outdoor():
        outdoor_K = np.array(
            [
                [223.9940010790056, 0, 170.7684322973841],
                [0, 223.61783486959376, 128.18711828338436],
                [0, 0, 1]
            ],
            dtype=np.float32,
        )
        resolution = (346, 260)
        return outdoor_K, resolution
    
    @staticmethod
    def load_distortion_outdoor():
        distortion_coeffs = [-0.033904378348448685, -0.01537260902537579, -0.022284741346941413, 0.0069204143687187645]
        return distortion_coeffs

if __name__ == "__main__":
    gt_path = "/home/rpg/Downloads/MVSEC/gt/outdoor_day1_gt.hdf5"
    data_path = "/home/rpg/Downloads/MVSEC/hdf5/outdoor_day1_data.hdf5"
    mvsec = MVSECLoader(data_path, gt_path)
    print(f"calibration: {mvsec.calibration}")
    print(f"image_shape: {mvsec.image_shape}")
    print(f"events: {mvsec.events}")
    print(f"events_ts: {mvsec.events_ts}")
    print(f"images: {mvsec.images}")
    print(f"images_ts: {mvsec.images_ts}")
    print(f"events_indices: {mvsec.events_indices}")
    print(f"T: {mvsec.T}")
    print(f"poses_ts: {mvsec.poses_ts}")
    print(f"depths: {mvsec.depths}")
    print(f"depths_ts: {mvsec.depths_ts}")
    print(f"flow: {mvsec.flow}")
    print(f"flow_ts: {mvsec.flow_ts}")
