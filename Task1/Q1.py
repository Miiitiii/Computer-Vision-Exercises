# importing the opencv module
import cv2

# using imread('path') and 1 denotes read as  color image
src = cv2.imread('MeeseeksHQ.png', 1)

# This is using for display the image
cv2.imshow('BGR', src)
cv2.waitKey()  # This is necessary to be required so that the image doesn't close immediately.

# Using cv2.cvtColor() method
# Using cv2.COLOR_BGR2GRAY color space
# conversion code
src2gry = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# Displaying the image
cv2.imshow('Gray', src2gry)

cv2.waitKey()  # This is necessary to be required so that the image doesn't close immediately.
# It will run continuously until the key press.
cv2.destroyAllWindows()
