# Define a function to map rover space pixels to world space
# The first task is to perform a rotation. You'll accomplish this by applying a
# rotation matrix(https://en.wikipedia.org/wiki/Rotation_matrix) to your rover
#space pixel values
import numpy as np

def rotate_pix(xpix, ypix, yaw):
    # yaw angle is recorded in degrees so first convert to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))

    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result
    return xpix_rotated, ypix_rotated
