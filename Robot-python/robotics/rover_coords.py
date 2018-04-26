# Define a function to convert from image coords to rover coords
'''
extract the pixel positions of all navigable terrain (white) pixels and then
transform those values to "rover-centric" coordinates, meaning a coordinate
frame where the rover camera is at (x, y) = (0, 0).
'''
import numpy as np
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the
    # center bottom of the image.
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel
