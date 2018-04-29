import cv2 # OpenCV for perspective transform
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.misc # For saving images as needed
import glob  # For reading in a list of images from a folder
import imageio

path = './IMG/*'
img_list = glob.glob(path)

print(img_list)
idx = np.random.randint(0, len(img_list)-1)
image = mpimg.imread(img_list[idx])

im1 = mpimg.imread('./IMG/example_grid1.jpg')
im2 = mpimg.imread('./IMG/example_grid2.jpg')
im3 = mpimg.imread('./IMG/example_rock1.jpg')
im4 = mpimg.imread('./IMG/example_rock2.jpg')
im5 = mpimg.imread('./IMG/map_bw.png')


# Do some plotting
fig = plt.figure(figsize=(12,9)) # width, height in inches.
plt.subplot(231) # Number of rows/columns of the subplot grid.
plt.imshow(im1)
plt.subplot(232)
plt.imshow(im2)
plt.subplot(233)
plt.imshow(im3, cmap='gray')
plt.subplot(234)
plt.imshow(im4, cmap='gray')
plt.subplot(235)
plt.imshow(im5, cmap='gray')

'''
plt.ylim(-160, 160)
plt.xlim(0, 160)
arrow_length = 100
x_arrow = arrow_length * np.cos(mean_dir)
y_arrow = arrow_length * np.sin(mean_dir)
plt.arrow(0, 0, x_arrow, y_arrow, color='red', zorder=2, head_width=10, width=2)
'''
#plt.imshow(image)
plt.show()
