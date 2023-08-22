# organize imports
import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
from time import time

# This will return video from the first webcam on your computer.
cap = cv2.VideoCapture(0)
#
# # Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter('Q2.avi', fourcc, 7.5, (640, 480))
ret, frame = cap.read()

old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
old_gray = cv2.GaussianBlur(old_gray, ksize=(15, 15), sigmaX=2, sigmaY=2)

feature_params = dict(
    maxCorners=10,
    qualityLevel=0.1,
    minDistance=120,
    blockSize=5
)

lk_params = dict(winSize=(20, 20),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

p1 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

endpoints1 = []
endpoints0 = []

start_time = time()
while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, ksize=(15, 15), sigmaX=2, sigmaY=2)

    p1, state, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p1, None, **lk_params)
    old_gray = frame_gray.copy()
    endpoints2 = []
    if len(p1) > 0:
        for i, s in enumerate(state):
            if s == 1:
                endpoints2.append(p1[i])
            else:
                endpoints2.append(None)
    if len(endpoints2) > 0 and len(endpoints1) > 0:
        for old, new in zip(endpoints1, endpoints2):
            if old is not None and new is not None:
                old = old[0].astype(int)
                new = new[0].astype(int)
                frame = cv2.arrowedLine(frame, old, new, (255, 0, 0), 3)
    if len(endpoints1) > 0 and len(endpoints0) > 0:
        for old, new in zip(endpoints0, endpoints1):
            if old is not None and new is not None:
                old = old[0].astype(int)
                new = new[0].astype(int)
                frame = cv2.arrowedLine(frame, old, new, (0, 255, 0), 3)
    endpoints0 = endpoints1.copy()
    endpoints1 = endpoints2.copy()
    cv2.imshow('result', frame)
    videoWriter.write(frame)

    # Wait for 'a' key to stop the program
    if time() - start_time >= 30:
        break
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# Close the window / Release webcam
cap.release()

# After we release our webcam, we also release the output
videoWriter.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
