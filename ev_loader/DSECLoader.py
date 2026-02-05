import torch
import glob
import cv2
import h5py
import yaml
import numpy as np
from tqdm import tqdm
import os.path as osp
from torchvision.io import read_image

from ev_loader.representations.representations import EventToStack_Numpy
from ev_loader.utils.disparity_to_depth import disparity_to_points3D
from ev_loader.utils.utils_quaternions import (
    translation_and_quaternion_to_pose, 
    poses_array_to_transformation_matrix
    )
from ev_loader.data.event_handle import  H5EventHandle


class DSECLoader():
    def __init__(self, 
                 images_path, 
                 events_path, 
                 K_path, 
                 disparity_path, 
                 disparity_ts_path, 
                 T_path, 
                 poses_ts_path
                ):
        """Load the DSEC dataset from the given paths
        Args:
            images_path (str): Path to the images folder
            events_path (str): Path to the events file
            tstamps_path (str): Path to the timestamps file
            K_path (str): Path to the intrinsics file
            disparity_path (str): Path to the disparity images
        """
        self.images_path = images_path
        self.events_path = events_path
        self.K_path = K_path
        self.disparity_path = disparity_path
        self.disparity_ts_path = disparity_ts_path
        self.T_path = T_path
        self.poses_ts_path = poses_ts_path

        self.calibration = self.import_intrinsics()
        self.image_shape = self.image_shape_from_K()
        self.width, self.height = self.image_shape[0], self.image_shape[1]
        self.fix_events(events_path=events_path, image_shape=self.image_shape)
        self.events = self.import_events()
        self.images = self.import_images()

        self.depths = self.disparities_to_depths()
        self.depths_ts = self.import_timestamps(tstamps_path=disparity_ts_path)
        self.depths_ts -= self.depths_ts[0]

        self.T = self.import_T()
        self.poses_ts = self.import_timestamps(tstamps_path=poses_ts_path)

    @staticmethod
    def add_h5events_h_and_w(events: h5py.File, width: int, height: int):
        if "width" in events["events"]:
            del events["events"]["width"]
        if "height" in events["events"]:
            del events["events"]["height"]
        events["events"].create_dataset("width", data=width)
        events["events"].create_dataset("height", data=height)
        return events

    @staticmethod
    def cast_dataset_to(handle, type_x = np.uint16, type_y = np.uint16, type_t = np.int64, type_p = np.int8):    
        data_type = handle["events"]["x"].dtype
        if data_type != type_x:
            print(f"Casting x to {type_x} from {data_type} ")
            a = handle["events"]["x"].astype(type_x)[:]
            del handle["events"]["x"]
            handle["events"].create_dataset("x", data=a)
            print(f"Data casted to {handle['events']['x'].dtype}")

        data_type = handle["events"]["y"].dtype
        if data_type != type_y:
            print(f"Casting y to {type_y} from {data_type} ")
            a = handle["events"]["y"].astype(type_y)[:]
            del handle["events"]["y"]
            handle["events"].create_dataset("y", data=a)
            print(f"Data casted to {handle['events']['y'].dtype}")
    
        data_type = handle["events"]["t"].dtype
        if data_type != type_t:
            print(f"Casting t to {type_t} from {data_type} ")
            a = handle["events"]["t"].astype(type_t)[:]
            del handle["events"]["t"]
            handle["events"].create_dataset("t", data=a)
            print(f"Data casted to {handle['events']['t'].dtype}")

        data_type = handle["events"]["p"].dtype
        if data_type != type_p:
            print(f"Casting p to {type_p} from {data_type} ")
            a = handle["events"]["p"].astype(type_p)[:]
            del handle["events"]["p"]
            handle["events"].create_dataset("p", data=a)
            print(f"Data casted to {handle['events']['p'].dtype}")

        return handle
    
    def fix_events(self, events_path, image_shape):
        events_mod = h5py.File(events_path, "r+")
        print("Adding width and height to events")
        events = self.add_h5events_h_and_w(events_mod, width=image_shape[0], height=image_shape[1])
        print("Casting dataset to int64")
        events = self.cast_dataset_to(events_mod, type_t=np.int64)
        events_mod.close()
        return events
    
    @staticmethod
    def stack_representation(events, num_event_bins=5):
        event_representation = EventToStack_Numpy(num_bins=num_event_bins)
        stack = torch.tensor(event_representation(events))
        return stack
    
    def events_from_indices(self, events: H5EventHandle, i_start, i_stop):
        event_blob = events.get_between_idx(i_start, i_stop)
        event_tensor = self.stack_representation(event_blob)
        return event_tensor
    
    @staticmethod
    def get_indices_corresponding_to_timestamps(events_ts, image_ts):
        print(f"Events first timestamp: {events_ts[0]} and Image first timestamp: {image_ts[0]}")
        time_vicinity = np.subtract.outer(events_ts, image_ts) ** 2
        corresponding_frame_indices = np.argmin(time_vicinity, axis=1)
        corresponding_events_indices = np.argmin(time_vicinity, axis=0)
        return corresponding_frame_indices, corresponding_events_indices
           
    def form_the_data_stream(self, images, events, intrinsics, image_ts):
        n_events_selected = 250000
        events_ts = events.t[::n_events_selected]
        
        frame_inds, events_inds = self.get_indices_corresponding_to_timestamps(events_ts, image_ts)
        n_events_voxels = len(events.t) // n_events_selected
        
        i1 = 0
        data_list = []
        for i in tqdm(range(n_events_voxels)):
            i0 = i1
            i1 += n_events_selected
            event_stack = self.events_from_indices(events=events, i_start=i0, i_stop=i1)
            image = images[frame_inds[i]]
            
            # cast image and events to float32
            image = image.float()
            event_stack = event_stack.float()
        
            data_list.append((image, event_stack, intrinsics, torch.tensor([True])))

        return data_list, events_ts
    
    def import_images(self):
        images_paths = osp.join(self.images_path, "*{}".format(".png"))
        images_files = sorted(glob.glob(images_paths))
        print(f"Found {len(images_files)} Images")
        images = [read_image(image_f) for image_f in tqdm(images_files)]
        return images
    
    def import_intrinsics(self):
        K = yaml.safe_load(open(self.K_path))
        camera_matrix = K["intrinsics"]["camRect0"]["camera_matrix"]
        fx, fy, cx, cy = zip(camera_matrix)
        intrinsics = np.array(
            [
                [fx[0], 0, cx[0]],
                [0, fy[0], cy[0]],
                [0, 0, 1]
            ],
            dtype=np.float32,
        )
        return intrinsics

    def image_shape_from_K(self):
        K = yaml.safe_load(open(self.K_path))
        image_shape = K["intrinsics"]["camRect0"]["resolution"]
        return image_shape
    
    def import_events(self):
        if not self.events_path.endswith(".h5"):
            raise ValueError("Events must be a single .h5 file")
        events = h5py.File(self.events_path, "r+")
        x = events["events"]["x"][:]
        y = events["events"]["y"][:]
        t = events["events"]["t"][:]
        p = events["events"]["p"][:]
        return (x, y, t, p)
    
    def import_timestamps(self, tstamps_path=None):
        timestamps = np.loadtxt(tstamps_path)
        return timestamps
    
    def import_disparity(self):
        disparity_paths = osp.join(self.disparity_path, "*{}".format(".png"))
        disparity_files = sorted(glob.glob(disparity_paths))
        print(f"Found {len(disparity_files)} Disparity Images")
        disparities = [read_image(disparity_f) for disparity_f in tqdm(disparity_files)]
        return disparities
    
    def disparities_to_depths(self):
        disparity_paths = osp.join(self.disparity_path, "*{}".format(".png"))
        disparity_files = sorted(glob.glob(disparity_paths))
        print(f"Found {len(disparity_files)} Disparity Images")
        depths = []
        for disparity_f in tqdm(disparity_files):
            points3D = disparity_to_points3D(disparity_f, self.K_path)
            depth = points3D[:,:,2]
            # where the depth is not defined it shoudl be nan
            depth[depth == np.inf] = np.nan 

            depths.append(depth)
        return np.array(depths)
    
    def import_T(self):
        poses = np.loadtxt(self.T_path)
        return poses_array_to_transformation_matrix(poses)
    
    def events_from_inds(self, ind_0, ind_1):
        x,y,t,p = self.events
        return np.stack((x[ind_0:ind_1],y[ind_0:ind_1],t[ind_0:ind_1],p[ind_0:ind_1]), axis=1)
    

if __name__ == "__main__":
    pass