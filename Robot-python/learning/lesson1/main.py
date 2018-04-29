import cv2 # OpenCV for perspective transform
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from color_thresh import color_thresh
path = 'pic1.jpg'


image = mpimg.imread(path)

red_threshold = 170
green_threshold = 170
blue_threshold = 170
rgb_threshold = (red_threshold, green_threshold, blue_threshold)
colorsel = color_thresh(image, rgb_thresh=rgb_threshold)



from perspect_transform import perspect_transform
# to a grid where each 10x10 pixel square represents 1 square meter
# The destination box will be 2*dst_size on each side
dst_size = 5
# Set a bottom offset to account for the fact that the bottom of the image
# is not the position of the rover but a bit in front of it
# this is just a rough guess, feel free to change it!
bottom_offset = 6
source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])
#warped, mask = perspect_transform(image, source, destination)
one, warped= perspect_transform(image, source, destination)

#f, axs = plt.subplots(2,2,figsize=(15,15))
#fig = plt.figure(figsize=(12,9))

blue = image[:,:,0]

fig, axes = plt.subplots(3, 3)

#fig = plt.figure(figsize=(24,24))
plt.subplot(221)
plt.imshow(image)
plt.subplot(222)
plt.imshow(colorsel)
#plt.subplot(221)
#plt.imshow(one)
#plt.subplot(222)
#plt.imshow(warped)
#plt.subplot(225)
#plt.imshow(blue)
plt.show()
