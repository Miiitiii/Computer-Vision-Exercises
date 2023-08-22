# organize imports
import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt

# This will return video from the first webcam on your computer.
cap = cv2.VideoCapture(0)
#
# # Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter('Q1.avi', fourcc, 20.0, (640, 480))
ret, frame = cap.read()

old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
old_gray = cv2.GaussianBlur(old_gray, ksize=(15, 15), sigmaX=2, sigmaY=2)

feature_params = dict(
    maxCorners=10,
    qualityLevel=0.135,
    minDistance=120,
    blockSize=5
)

p1 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# corners = np.intp(p1)
#
# for i in corners:
#     x, y = i.ravel()
#     cv2.circle(old_gray, (x, y), 3, 255, 3)
#
# plt.imshow(old_gray, cmap='gray')
# plt.show()

while True:
    # reads frames from a camera
    # ret checks return at each frame
    ret, frame = cap.read()

    # output the frame
    videoWriter.write(frame)

    # The original input frame is shown in the window
    cv2.imshow('Original', frame)

    # Wait for 'a' key to stop the program
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# Close the window / Release webcam
cap.release()

# After we release our webcam, we also release the output
videoWriter.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
