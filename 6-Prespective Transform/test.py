#import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
example_grid = './example_grid1.jpg'
image = mpimg.imread(example_grid)
print(image.shape[1])
plt.imshow(image)
