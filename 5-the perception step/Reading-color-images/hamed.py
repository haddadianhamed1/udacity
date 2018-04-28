import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image
# There are six more images available for reading
# called sample1-6.jpg, feel free to experiment with the others!
image_name = 'sample.jpg'
image = mpimg.imread(image_name)
# Import the "numpy" package for working with arrays
#print(image.dtype, image.shape, np.min(image), np.max(image))
#print (image)
#for pix in image:
#     (pix[0][0])
#print(image[:,:,0])
#print(image[:,:,1])
#print(image[:,:,2])

# uint8 (160, 320, 3) 0 255
#plt.imshow(image)
#plt.show()

red_channel = np.copy(image)
#print (red_channel[0][1])
#print (red_channel[159][319])
#print (red_channel.shape[0])

for row in range(red_channel.shape[0]):
    for col in range(red_channel.shape[1]):
        a = (red_channel[row][col])
        for i in range(3):
            if (a[i] > 160):
                a[i] = 255
            else:
                a[i] = 0
        #print (a[0:3])
#red_channel[:,:,[0,1,2]] = 0 # Zero out the green and blue channels

red_channel1 = np.copy(image)
print (red_channel1[:,:,1])
red_channel2 = np.zeros_like(red_channel1[:,:,2])
if [(red_channel1[:,:,0] > 160) \
                & (red_channel1[:,:,1] > 160) \
                & (red_channel1[:,:,2] > 160)]:
    red_channel1[:,:,[0,1,2]] = 255
else:
    red_channel1[:,:,[0,1,2]] = 0
#print (red_channel1)
#print (red_channel1[:,:,1])
test = np.copy(image)
test[:,:,[2]]
fig = plt.figure(figsize=(12,3)) # Create a figure for plotting
plt.subplot(131) # Initialize subplot number 1 in a figure that is 3 columns 1 row
plt.imshow(red_channel) # Plot the red channel
plt.subplot(132) # Initialize subplot number 2 in a figure that is 3 columns 1 row
plt.imshow(red_channel1)  # Plot the green channel
plt.show()
