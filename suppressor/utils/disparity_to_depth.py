import cv2
import yaml
import glob
import numpy as np
import os.path as osp
from cv2 import reprojectImageTo3D, imread



def disparity_to_points3D(disparity_image_path, Q_image_path):
    Q = import_Q(Q_image_path)
    disparity = import_disparity(disparity_image_path)
    return reprojectImageTo3D(disparity, Q)

def import_Q(Q_image_path):
    K = yaml.safe_load(open(Q_image_path))
    Q = K["disparity_to_depth"]["cams_12"]
    return np.array(Q)

def import_disparity(disparity_image_path):
    disparity_image = imread(disparity_image_path,  cv2.IMREAD_GRAYSCALE)
    disparity_image.astype('float32')/256
    return disparity_image

if __name__ == "__main__":
    disparity_image_abs_path = "/home/rpg/Downloads/DSEC/interlaken_00_c/disparity_images"
    Q_image_path = "/home/rpg/Downloads/DSEC/interlaken_00_c/cam_to_cam.yaml"

    images_paths = osp.join(disparity_image_abs_path, "*{}".format(".png"))
    images_files = sorted(glob.glob(images_paths))
    depths = []
    for i, image_file in enumerate(images_files):
        depth = disparity_to_points3D(images_files[i], Q_image_path)
        depth_ = depth[:,:,2]
        depth_[depth_ == np.inf] = 0
        depths.append(depth_)
        # plt.clf()
        # plt.imshow(depth[:,:,2])
        # plt.savefig("/home/rpg/Desktop/event_suppression/plots/{}.png".format(str(i).zfill(4)))
    np.save("/home/rpg/Downloads/DSEC/interlaken_00_c/depth.npy", np.array(depths))
