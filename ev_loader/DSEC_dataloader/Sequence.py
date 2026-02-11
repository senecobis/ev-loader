"""
Modification of the official DSEC data loader https://github.com/uzh-rpg/DSEC/tree/main 
from Roberto Pellerito rpellerito@ifi.uzh.ch
"""

import cv2
import h5py
import torch
import yaml
import weakref
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from ..representations.representations import VoxelGrid, MeanTimestamps, EventToStack_Numpy
from ..utils.eventslicer import EventSlicer
from ..utils.utils_flow import load_flow


class Sequence(Dataset):
    # This class assumes the following structure in a sequence directory:
    #
    # seq_name (e.g. zurich_city_11_a)
    # ├── disparity
    # │   ├── event
    # │   │   ├── 000000.png
    # │   │   └── ...
    # │   └── timestamps.txt
    # └── events
    #     ├── left
    #     │   ├── events.h5
    #     │   └── rectify_map.h5
    #     └── right
    #         ├── events.h5
    #         └── rectify_map.h5

    def __init__(self, 
                seq_path: Path, 
                mode: str='train', 
                delta_t_ms: int=50, 
                num_bins: int=15, 
                representation: str='voxel',
                load_opt_flow: bool=True
                ):
        assert num_bins >= 1
        assert delta_t_ms <= 100, 'adapt this code, if duration is higher than 100 ms'
        assert seq_path.is_dir()
        available_representations = ['voxel', 'tstamp_image', 'voxel_tstamp_image', 'raw', 'stack']
        assert representation in available_representations, f"Unkown representation from {available_representations}"

        # NOTE: Adapt this code according to the present mode (e.g. train, val or test).
        self.mode = mode
        self.sequence_path = seq_path
        self.sequence_id = seq_path.stem
        self.representation = representation
        self.load_opt_flow = load_opt_flow

        # Save output dimensions
        self.height = 480
        self.width = 640
        self.num_bins = num_bins
        self.total_pixels_num = self.height * self.width

        # Set event representation
        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=True)
        self.mean_tstamp = MeanTimestamps(self.height, self.width, smooth=False)
        self.stack = EventToStack_Numpy(num_bins=self.num_bins, height=self.height, width=self.width)

        # Save delta timestamp in ms
        self.delta_t_ms = delta_t_ms
        self.delta_t_us = delta_t_ms * 1000

        # load disparity timestamps
        disp_dir = seq_path / 'disparity'
        assert disp_dir.is_dir()
        self.timestamps = np.loadtxt(disp_dir / 'timestamps.txt', dtype='int64')

        # load events disparity paths
        ev_disp_dir = disp_dir / 'event'
        self.disp_gt_pathstrings = self.import_image_like_paths(ev_disp_dir)
        # Check if the number of disparity maps and timestamps match
        assert len(self.disp_gt_pathstrings) == self.timestamps.size

        # Assume disparity is at the same rate of every GT
        # Get the FPS of the GT suppose the timestamp are in us
        self.FPS_GT = round(1e6 / np.mean(np.diff(self.timestamps)))

        # Load intrinsics
        self.K_path = seq_path / 'calibration/cam_to_cam.yaml'
        assert self.K_path.is_file()
        self.intrinsics = yaml.safe_load(open(self.K_path))
        self.Q = np.array(self.intrinsics["disparity_to_depth"]["cams_12"])
        self.K = np.array(self.intrinsics["intrinsics"]["cam0"]["camera_matrix"])

        # Load dynamic objects masks
        self.dyn_mask_path = seq_path / "mask/event_mask"
        self.dyn_masks_pathstrings = self.import_image_like_paths(self.dyn_mask_path)
        # Check if the number of dynamic masks and disparity GTs match
        if not len(self.dyn_masks_pathstrings) == len(self.disp_gt_pathstrings):
            raise ValueError("Number of dynamic masks and disparity GTs do not match for {}".format(seq_path))

        # Remove first disparity path and corresponding timestamp.
        # This is necessary because we do not have events before the first disparity map.
        assert int(Path(self.disp_gt_pathstrings[0]).stem) == 0
        self.disp_gt_pathstrings.pop(0)
        self.timestamps = self.timestamps[1:]

        # Remove the first event mask path.
        # This is necessary because we do not have events before the first dynamic mask.
        assert int(Path(self.dyn_masks_pathstrings[0]).stem) == 0
        self.dyn_masks_pathstrings.pop(0)

        self.h5f = dict()
        self.rectify_ev_maps = dict()
        self.event_slicers = dict()

        self.locations = ['left', 'right']
        ev_dir = seq_path / 'events'
        for location in self.locations:
            ev_dir_location = ev_dir / location
            ev_data_file = ev_dir_location / 'events.h5'
            ev_rect_file = ev_dir_location / 'rectify_map.h5'

            h5f_location = h5py.File(str(ev_data_file), 'r')
            self.h5f[location] = h5f_location
            self.event_slicers[location] = EventSlicer(h5f_location)
            with h5py.File(str(ev_rect_file), 'r') as h5_rect:
                self.rectify_ev_maps[location] = h5_rect['rectify_map'][()]

        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

        # get images from DSEC dataset
        self.frames_path = seq_path / 'images/left/rectified'
        self.frames_pathstrings_original = self.import_image_like_paths(self.frames_path)
        # TODO load all images, for now we subselect to have the same number of images as GT
        self.frames_pathstrings = self.frames_pathstrings_original[::2]
        # Remove the first image as we do not have events before the first data-point
        self.frames_pathstrings.pop(0)
        if len(self.disp_gt_pathstrings) != len(self.frames_pathstrings):
            raise ValueError("Number of disparity GTs and images do not match for {}".format(seq_path))

        # get optical flow from DSEC dataset
        self.optical_flow_path = seq_path / 'flow'
        self.forward_flow_path = self.optical_flow_path / 'forward' # flow from t_i to t_{i+1}
        self.backward_flow_path = self.optical_flow_path / 'backward' # flow from t_{i-1} to t_i
        self.flow_exists = self.load_opt_flow and self.forward_flow_path.is_dir() and self.backward_flow_path.is_dir()
        if not self.flow_exists:
            # print(f"Optical flow not found for {seq_path} -> Not loading the optical flow for this sequence")
            pass
        else:
            self.timestamps_flow = np.loadtxt(self.optical_flow_path / 'timestamps.txt', dtype='int64')
            self.forward_flow_pathstrings = self.import_image_like_paths(self.forward_flow_path)
            self.backward_flow_pathstrings = self.import_image_like_paths(self.backward_flow_path)

    @staticmethod
    def import_image_like_paths(data_path: Path):
        assert data_path.is_dir()
        datas = list()
        for entry in data_path.iterdir():
            assert entry.is_file()
            assert str(entry.name).endswith('.png')
            datas.append(str(entry))
        datas.sort()
        return datas
    
    @staticmethod
    def events_to_voxel_grid(voxel_grid, x, y, p, t):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        return voxel_grid.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t))

    def _events_to_voxel_grid(self, x, y, p, t):
        return self.events_to_voxel_grid(self.voxel_grid, x, y, p, t)
    
    
    def events_to_mean_tstamp(self, x, y, t):
        if not isinstance(t, torch.Tensor):
            x = torch.from_numpy(x.astype(np.int64))
            y = torch.from_numpy(y.astype(np.int64))
            t = torch.from_numpy(t.astype(np.float32))
        return self.mean_tstamp.convert(x=x, y=y, t=t)
    
    @staticmethod
    def get_disparity_map(filepath: Path):
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        normalised_disp = disp_16bit.astype('float32')/256
        return normalised_disp
    
    @staticmethod
    def get_dynamic_obj_mask(filepath: Path):
        mask = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
        mask = mask / 255
        return mask
    
    @staticmethod
    def get_frame(filepath: str):
        return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

    @staticmethod
    def close_callback(h5f_dict):
        for k, h5f in h5f_dict.items():
            h5f.close()

    def rectify_events(self, x: np.ndarray, y: np.ndarray, location: str):
        assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_maps[location]
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]
    
    @staticmethod
    def disparities_to_depths(disparity, Q):
        disparity = np.array(disparity)
        points3D = cv2.reprojectImageTo3D(disparity, Q)
        depth = points3D[:,:,2]
        depth[depth == np.inf] = np.nan 
        return depth
    
    def get_events_start_end_time(self, ts_start, ts_end):
        event_data = self.event_slicers["left"].get_events(ts_start, ts_end)

        p = event_data['p']
        t = event_data['t']
        x = event_data['x']
        y = event_data['y']
        return x, y, p, t
    
    def get_rectified_events(self, x: np.ndarray, y: np.ndarray):
        xy_rect = self.rectify_events(x, y, location="left")
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]
        return x_rect, y_rect

    def get_rectified_events_start_end_time(self, ts_start, ts_end):
        x,y,p,t = self.get_events_start_end_time(ts_start, ts_end)

        # TODO the rectification map introduce a checkerboard pattern in the events
        # This might be a problem for the network learning from event representation
        x_rect, y_rect = self.get_rectified_events(x, y)
        return x_rect, y_rect, p, t
    
    def get_event_representation(self, x, y, p, t):
        if self.representation == 'voxel':
            event_representation = self._events_to_voxel_grid(x, y, p, t)
        elif self.representation == 'stack':
            event_representation = self.stack(x, y, p)
        elif self.representation == 'raw':
            event_representation = np.stack([x, y, p, t], axis=0)
        elif self.representation == 'tstamp_image':
            event_representation = self.events_to_mean_tstamp(x, y, t)
        elif self.representation == 'voxel_tstamp_image':
            voxel_grid = self._events_to_voxel_grid(x, y, p, t)
            mean_tstamp_img = self.events_to_mean_tstamp(x, y, t)
            event_representation = np.concatenate([voxel_grid, mean_tstamp_img], axis=0)
        
        if not isinstance(event_representation, torch.Tensor):
            event_representation = torch.from_numpy(event_representation).float()
        return event_representation
    
    def __len__(self):
        return len(self.disp_gt_pathstrings)
    
    def disparity_gt(self, index):
        disp_gt_path = Path(self.disp_gt_pathstrings[index])
        disparity = self.get_disparity_map(disp_gt_path)
        return torch.from_numpy(disparity).float()
    
    def dynamic_mask_gt(self, index):
        dyn_mask_gt_path = Path(self.dyn_masks_pathstrings[index])
        dynamic_mask = self.get_dynamic_obj_mask(dyn_mask_gt_path)
        return torch.from_numpy(dynamic_mask).float().unsqueeze(0)
    
    def depth_gt(self, disparity):
        depth = self.disparities_to_depths(disparity=disparity, Q=self.Q)
        return torch.from_numpy(depth).float()
    
    def frame_gt(self, index):
        frame_path = self.frames_pathstrings[index]
        frame = self.get_frame(frame_path)
        return torch.from_numpy(frame).permute(2, 0, 1).float()
    
    def file_index(self, index):
        disp_gt_path = Path(self.disp_gt_pathstrings[index])
        return int(disp_gt_path.stem)
    
    def forward_flow_gt(self, index):
        flow_path = Path(self.forward_flow_pathstrings[index])
        flow = load_flow(flow_path)
        return flow
    
    def backward_flow_gt(self, index):
        flow_path = Path(self.backward_flow_pathstrings[index])
        flow = load_flow(flow_path)
        return flow

    def __getitem__(self, index):
        ts_end = self.timestamps[index]
        ts_start = ts_end - self.delta_t_us

        file_index = self.file_index(index)
        disparity = self.disparity_gt(index)
        dynamic_mask = self.dynamic_mask_gt(index)
        depth = self.depth_gt(disparity)
        frame = self.frame_gt(index)
        
        x_rect, y_rect, p, t = self.get_rectified_events_start_end_time(ts_start, ts_end)
        self.x_rect = x_rect
        self.y_rect = y_rect
        self.p = p
        self.t = t
        event_representation = self.get_event_representation(x_rect, y_rect, p, t)

        output = {
            'file_index': file_index,
            'sequence_id': self.sequence_id,
            'disparity_gt': disparity,
            'dynamic_mask_gt': dynamic_mask,
            'depth_gt': depth,
            'intrinsics': self.K,
            'representation': {"left": event_representation},
            'frame': frame,
            'is_GT_available': True
        }

        if self.flow_exists and index < len(self.forward_flow_pathstrings):
            forward_flow = self.forward_flow_gt(index)
            backward_flow = self.backward_flow_gt(index)
            output['forward_flow_gt'] = forward_flow
            output['backward_flow_gt'] = backward_flow
        return output
    

if __name__ == '__main__':
    # seq_abs_path = Path("/data/scratch/pellerito/datasets/DSEC/train/interlaken_00_c")
    seq_abs_path = Path("/home/rpg/Downloads/DSEC/train/thun_00_a")
    dsec_seq = Sequence(seq_path=seq_abs_path, num_bins=2, representation="stack")
    # loader_outputs = dsec_seq[0]
    for loader_outputs in dsec_seq:
        file_index = loader_outputs['file_index']
        sequence_id = loader_outputs['sequence_id']
        disparity_gt = loader_outputs['disparity_gt']
        dynamic_mask_gt = loader_outputs['dynamic_mask_gt']
        depth_gt = loader_outputs['depth_gt']
        intrinsics = loader_outputs['intrinsics']
        frame = loader_outputs['frame']
        event_representation = loader_outputs['representation']["left"]

        forward_flow_gt = loader_outputs["forward_flow_gt"]
        backward_flow_gt = loader_outputs["backward_flow_gt"]

        print(f"File index: {file_index}")
        print(f"Sequence ID: {sequence_id}")
        print(f"Disparity GT shape: {disparity_gt.shape}")
        print(f"Dynamic mask GT shape: {dynamic_mask_gt.shape}")
        print(f"Depth GT shape: {depth_gt.shape}")
        print(f"Intrinsics: {intrinsics}")
        print(f"frame shape: {frame.shape}")
        print(f"Event representation shape: {event_representation.shape}")
        print(f"Forward flow GT shape: {forward_flow_gt.shape}")
        print(f"Backward flow GT shape: {backward_flow_gt.shape}")