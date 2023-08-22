import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('hcont.jpg', 1)

cv2.imshow("main_image", img)
cv2.waitKey()

def histogram_equalization(img_in):
    # segregate color streams
    b, g, r = cv2.split(img_in)
    color = ('b', 'g', 'r')
    fig, ax = plt.subplots()
    for i, col in enumerate(color):
        histr = cv2.calcHist([img_in], [i], None, [256], [0, 256])
        ax.plot(histr, color=col)
        ax.set_xlim([0, 256])
    fig.show()
    fig.savefig('histogram_image_before.png')
    plt.close(fig)

    equ_b = cv2.equalizeHist(b)
    equ_g = cv2.equalizeHist(g)
    equ_r = cv2.equalizeHist(r)
    img_out = cv2.merge((equ_b, equ_g, equ_r))
    fig1, ax1 = plt.subplots()
    for i, col in enumerate(color):
        histr = cv2.calcHist([img_out], [i], None, [256], [0, 256])
        ax1.plot(histr, color=col)
        ax1.set_xlim([0, 256])
    fig1.show()
    fig1.savefig('histogram_image_after.png')
    plt.close(fig1)
    return img_out


img_out = histogram_equalization(img)
cv2.imshow("new_image", img_out)
cv2.waitKey()
# Filename
filename = 'hcont_equ.jpg'

# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, img_out)
