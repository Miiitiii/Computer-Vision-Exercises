# importing the opencv module
import cv2
import numpy as np

# using imread('path') and 1 denotes read as  color image
img = cv2.imread('edge.jpg', 1)

cv2.imshow("edge_image", img)
cv2.waitKey()

edges = cv2.Canny(img, 200, 400, True)
cv2.imshow("Edge Detected Image", edges)
cv2.imshow("Original Image", img)
cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image

# Filename
filename = 'edge_canny_before.jpg'

# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, edges)

img_median = cv2.medianBlur(img, 5)
cv2.imshow("new", img_median)
cv2.waitKey(0)  # waits until a key is pressed

edges = cv2.Canny(img_median, 200, 400, True)
cv2.imshow("Edge Detected Image", edges)
cv2.imshow("Original Image", img_median)
cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image
# Filename
filename = 'edge_canny_after_median.jpg'

# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, edges)

# Filename
filename = 'image_median.jpg'

# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, img_median)

img_gauss = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow("new", img_gauss)
cv2.waitKey(0)  # waits until a key is pressed

edges = cv2.Canny(img_gauss, 200, 400, True)
cv2.imshow("Edge Detected Image", edges)
cv2.imshow("Original Image", img_gauss)
cv2.waitKey(0)  # waits until a key is pressed
# Filename
filename = 'edge_canny_after_gauss.jpg'

# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, edges)

# Filename
filename = 'image_gauss.jpg'

# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, img_gauss)
