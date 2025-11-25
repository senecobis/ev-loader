import os
import torch
import collections
import numpy as np
import rerun as rr
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from suppressor.utils.utils_quaternions import translation_and_quaternion_to_pose
from suppressor.utils.utils import rerun_log_pose_image
from suppressor.DSEC_dataloader.sequence import Sequence
from suppressor.utils.representations import VoxelGrid

import matplotlib.pyplot as plt


class Evimo2Loader():
    def __init__(self, dataset_path, sequence_name, num_bins, events_source="samsung"):
        """Load the Evimo2 dataset and provide access to the data.

        Args:
            dataset_path (str): Path to the dataset.
            sequence_name (str): Name of the sequence.
            events_source (str, optional): Source of the events. Defaults to "left_cam".

        Raises:
            ValueError: events_source must be one of ["left_cam", "samsung", "flea"].
        """
        self.dataset_path = Path(dataset_path)
        self.sequence_name = sequence_name
        self.file_flea = self.dataset_path / "npz_flea3_7_imo/flea3_7/imo/eval" / self.sequence_name
        self.file_left_cam = self.dataset_path / "left_camera/imo/eval" / self.sequence_name
        self.file_samsung = self.dataset_path / "npz_samsung_mono_imo/samsung_mono/imo/eval" / self.sequence_name
        self.file_extrinsics = self.dataset_path / "left_camera/imo/eval" / self.sequence_name / "dataset_extrinsics.npz"

        self.flea = self.load_flea()
        self.left_cam = self.load_left_cam()
        self.samsung = self.load_samsung()
        self.extrinsics = self.load_extrinsic_calibration()
        
        self.event_sensors = ["left_cam", "samsung", "flea"]
        if events_source == "left_cam":
            self.sensor = self.left_cam
            raise ValueError("left_cam is not supported yet, no flow is available in left_cam")
        elif events_source == "samsung":
            self.sensor = self.samsung
        elif events_source == "flea":
            self.sensor = self.flea
        else:        
            raise ValueError(f"events_source must be one of {self.event_sensors}")
        
        self.calibration = self.get_calibration(self.sensor)
        self.flow, self.flow_ts = self.get_flow(self.sensor)
        self.events, self.image_shape = self.events_and_shape_from_struct(sensor_struct=self.sensor)
        self.T, self.poses_ts = self.get_T_from_sensor(sensor_struct=self.sensor)
        self.depths = self.get_ordered_depths(sensor_struct=self.sensor)
        self.masks = self.get_segmentation_masks(sensor_struct=self.sensor)

        self.first_event_tstamp = self.events[0, 2]
        self.last_event_tstamp = self.events[-1, 2]

        self.num_bins = num_bins
        self.width, self.height = self.image_shape
        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=True)

        # get mask ids
        self.object_ids = np.unique(self.masks)
        self.moving_object_ids_mask = np.ones_like(self.object_ids)


    def load_flea(self):
        file = self.file_flea
        classical = np.load(os.path.join(file, 'dataset_classical.npz'))
        depth = np.load(os.path.join(file, 'dataset_depth.npz'))
        events_p = np.load(os.path.join(file, 'dataset_events_p.npy'))
        events_t = np.load(os.path.join(file, 'dataset_events_t.npy'), allow_pickle=True)
        events_xy = np.load(os.path.join(file, 'dataset_events_xy.npy'))
        meta  = np.load(os.path.join(file, 'dataset_info.npz'), allow_pickle=True)['meta'].item()
        mask  = np.load(os.path.join(file, 'dataset_mask.npz'))
        data = {
            'meta': meta,
            'depth': depth,
            'mask': mask,
            'classical': classical,
            'events_xy': events_xy,
            'events_p': events_p,
            'events_t': events_t,
        }
        return data

    def load_left_cam(self):
        file = self.file_left_cam
        meta  = np.load(os.path.join(file, 'dataset_info.npz'), allow_pickle=True)['meta'].item()
        depth = np.load(os.path.join(file, 'dataset_depth.npz'))
        mask  = np.load(os.path.join(file, 'dataset_mask.npz'))
        events_xy = np.load(os.path.join(file, 'dataset_events_xy.npy'))
        events_p = np.load(os.path.join(file, 'dataset_events_p.npy'))
        events_t = np.load(os.path.join(file, 'dataset_events_t.npy'), allow_pickle=True)
        events_t = events_t.astype(np.float32)
        data = {
            'meta': meta,
            'depth': depth,
            'mask': mask,
            'events_xy': events_xy,
            'events_p': events_p,
            'events_t': events_t
        }
        return data

    def load_samsung(self):
        file = self.file_samsung
        meta  = np.load(os.path.join(file, 'dataset_info.npz'), allow_pickle=True)['meta'].item()
        depth = np.load(os.path.join(file, 'dataset_depth.npz'))
        mask  = np.load(os.path.join(file, 'dataset_mask.npz'))
        events_xy = np.load(os.path.join(file, 'dataset_events_xy.npy'))
        events_p = np.load(os.path.join(file, 'dataset_events_p.npy'))
        events_t = np.load(os.path.join(file, 'dataset_events_t.npy'), allow_pickle=True).astype(np.float32)
        flow = np.load(os.path.join(file, 'dataset_flow.npz'))
        data = {
            'meta': meta,
            'depth': depth,
            'mask': mask,
            'events_xy': events_xy,
            'events_p': events_p,
            'events_t': events_t,
            'flow': flow
        }
        return data
    
    @staticmethod
    def events_and_shape_from_struct(sensor_struct):
        events_np = np.concatenate((sensor_struct['events_xy'], sensor_struct['events_t'][..., None],  sensor_struct['events_p'][..., None]), axis=1)
        events_np[..., -1][events_np[..., -1] == 0] = -1
        image_shape = [sensor_struct['meta']['meta']['res_x'], sensor_struct['meta']['meta']['res_y']]

        print(f"event time: {sensor_struct['events_t']}")
        print(f"first event xy: {sensor_struct['events_xy'][0]}")
        print(f"image_shape: {image_shape} and events.shape: {events_np.shape}")

        events_np[:, 3] = (events_np[:, 3] +1)/2

        return events_np, image_shape
    
    @staticmethod
    def get_calibration(sensor_struct):
        calibration = np.zeros((3, 3))
        calibration[0, 0] = sensor_struct["meta"]["meta"]["fx"]
        calibration[1, 1] = sensor_struct["meta"]["meta"]["fy"]
        calibration[0, 2] = sensor_struct["meta"]["meta"]["cx"]
        calibration[1, 2] = sensor_struct["meta"]["meta"]["cy"]
        calibration[2, 2] = 1.0
        return calibration

    def load_extrinsic_calibration(self):
        path = self.file_extrinsics
        data = np.load(path, allow_pickle=True)
        tx = data["t_rigcamera"].item()["x"]
        ty = data["t_rigcamera"].item()["y"]
        tz = data["t_rigcamera"].item()["z"]
        qx = data["q_rigcamera"].item()["x"]
        qy = data["q_rigcamera"].item()["y"]
        qz = data["q_rigcamera"].item()["z"]
        qw = data["q_rigcamera"].item()["w"]
        T_rigcamera = translation_and_quaternion_to_pose([tx, ty, tz], [qx, qy, qz, qw])
        return T_rigcamera

    @staticmethod
    def get_flow(sensor_struct):
        struct = sensor_struct["flow"]
        flow = []
        for k, v in struct.items():
            if "flow" in k:
                flow.append(v)
        t = struct["t"]
        flow = np.stack(flow)
        flow = flow.transpose(0, 3, 1, 2)
        flow = np.flip(flow, axis=1)
        return flow, t

    @staticmethod
    def get_pose(sensor_struct):
        translations = []
        quaternions = []
        timestamps = []
        for i in range(len(sensor_struct["meta"]["frames"])):
            if not "cam" in sensor_struct["meta"]["frames"][i]:
                continue
            t = sensor_struct["meta"]["frames"][i]["cam"]["pos"]["t"]
            xyz = np.stack((t["x"], t["y"], t["z"]))
            translations.append(xyz)

            q = sensor_struct["meta"]["frames"][i]["cam"]["pos"]["q"]
            q_ = np.stack((q["w"], q["x"], q["y"], q["z"]))
            quaternions.append(q_)

            ts = sensor_struct["meta"]["frames"][i]["ts"]
            timestamps.append(ts)
        translations = np.stack(translations)
        quaternions = np.stack(quaternions)
        poses_ts = np.stack(timestamps)
        return translations, quaternions, poses_ts
    
    @staticmethod
    def get_T_from_t_and_q(t, q):
        Rot_matrix = R.from_quat(q).as_matrix()
        T = np.tile(np.eye(4), (len(Rot_matrix), 1, 1))
        T[:, :3, :3] = Rot_matrix
        T[:, :3, 3] = t
        return T
    
    def get_T_from_sensor(self, sensor_struct):
        translations, quaternions, poses_ts = self.get_pose(sensor_struct)
        T = self.get_T_from_t_and_q(t=translations, q=quaternions)
        return T, poses_ts
    
    def get_ordered_depths(self, sensor_struct):
        depth_d = collections.OrderedDict(sorted(sensor_struct["depth"].items()))
        depth = np.stack(list(depth_d.values()))
        return depth
    
    def plot_T(self):
        # Log the absolute poses using rerun_log_pose_as_t_and_quaterions
        rr.init("rerun_example_pinhole_perspective", spawn=True)
        for i, T_ in enumerate(self.T):
            rerun_log_pose_image(f"trajectory/frame{i}", T_, image=self.depths[i])

    def get_segmentation_masks(self, sensor_struct):
        masks = collections.OrderedDict(sorted(sensor_struct["mask"].items()))
        st_masks = np.stack(list(masks.values()))
        return st_masks
    
    def __len__(self):
        """Get the length of the dataset."""
        return len(self.poses_ts)
    
    def get_events_by_time(self, ts_0, ts_1):
        """Get events by time."""
        if ts_0 >= ts_1:
            raise ValueError("ts_0 should be less than ts_1")
        t_events = self.events[:, 2]
        if ts_0 < self.first_event_tstamp:
            ind_0 = 0
        else:
            ind_0 = np.argmin(np.abs(t_events - ts_0))

        if ts_1 > self.last_event_tstamp:
            ind_1 = len(t_events) - 1
        else:
            ind_1 = np.argmin(np.abs(t_events - ts_1))
        return self.events[ind_0:ind_1]
    
    def _events_to_voxel_grid(self, x, y, p, t):
        return Sequence.events_to_voxel_grid(self.voxel_grid, x, y, p, t)
    
    def __getitem__(self, index):
        """Get item by index."""
        if index >= len(self) or index == 0:
            return None
        
        ts_1 = self.poses_ts[index]
        ts_0 = self.poses_ts[index - 1]
        events = self.get_events_by_time(ts_0, ts_1)
        voxel = self._events_to_voxel_grid(x=events[:, 0], y=events[:, 1], p=events[:, 3], t=events[:, 2])

        flow = torch.from_numpy(self.flow[index].copy())
        mask = torch.from_numpy(self.masks[index].copy())
        data = {
            "voxel": voxel.unsqueeze(0),
            "flow": flow,
            "mask": mask,
        }
        return data
    

if __name__ == "__main__":
    evimo = Evimo2Loader(
        dataset_path="/home/rpg/Downloads/EVMIO_dataset",
        sequence_name="scene13_dyn_test_00_000000",
        events_source="samsung",
        num_bins=2
    )
    item_ = evimo[100]
    print(item_)