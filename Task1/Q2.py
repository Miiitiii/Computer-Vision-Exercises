# importing the opencv module
import cv2
import numpy as np

# using imread('path') and 1 denotes read as  color image
img = cv2.imread('mr meeskees.png', 1)

# This is using for display the image
cv2.imshow('BGR', img)
cv2.waitKey()  # This is necessary to be required so that the image doesn't close immediately.
# It will run continuously until the key press.
# cv2.destroyAllWindows()


# first method
# define the boundaries
lower = np.array([100, 80, 50])
upper = np.array([150, 170, 120])
# find the colors within the specified boundaries and apply
# the mask
mask = cv2.inRange(img, lower, upper)

# define kernel size
kernel = np.ones((7, 7), np.uint8)
# Remove unnecessary noise from mask
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

segmented_img = cv2.bitwise_and(img, img, mask=mask)
# Find contours from the mask
contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = cv2.drawContours(segmented_img, contours, -1, (0, 0, 255), 3)
# Showing the output
# show the images
cv2.imshow("images", output)
cv2.waitKey(0)
# Filename
filename = 'first_method.jpg'

# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, output)

# second method
# convert to hsv colorspace
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower bound and upper bound for Green color
lower_bound = np.array([50, 60, 70])
upper_bound = np.array([100, 255, 255])

# find the colors within the boundaries
mask = cv2.inRange(hsv, lower_bound, upper_bound)

# define kernel size
kernel = np.ones((7, 7), np.uint8)

# Remove unnecessary noise from mask
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Segment only the detected region
segmented_img = cv2.bitwise_and(img, img, mask=mask)

# Find contours from the mask
contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# output = cv2.drawContours(segmented_img, contours, -1, (0, 0, 255), 3)
# Draw contour on original image
output = cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
# Showing the output
cv2.imshow("Output", output)
cv2.waitKey(0)

# Filename
filename = 'second_method.jpg'

# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, output)
