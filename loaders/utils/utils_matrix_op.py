import numpy as np


def coordinates_to_image(coordinates, magnitudes, image_shape):
    """ This function takes the coordinates of the magnitued values and put it a 3D matrix
        with size H X W X 1 where the values are the magnitudes
    """
    width, height = image_shape
    magnitude_image = np.zeros((height, width))
    magnitude_image[coordinates[:, 1], coordinates[:, 0]] = magnitudes
    return magnitude_image


def coordinates_flow3D(start_coords, end_coords, image_shape):
    """ This function takes the start and end coordinates of the flow and returns the 3D flow matrix
    """
    flatten_flow = end_coords - start_coords
    start_coords = start_coords.astype(int)

    # Vectorized version of getting the segmentation mask
    x_2D = coordinates_to_image(start_coords, flatten_flow[:,0], image_shape)
    y_2D = coordinates_to_image(start_coords, flatten_flow[:,1], image_shape)
    flow_3D = np.stack((x_2D, y_2D), axis=0)
    return flow_3D

def magnitude_of(start_coords, end_coords):
    dx = end_coords[:, 0] - start_coords[:, 0]
    dy = end_coords[:, 1] - start_coords[:, 1]

    magnitudes = np.sqrt(dx**2 + dy**2)
    return magnitudes
