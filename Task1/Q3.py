# importing the opencv module
import cv2
import numpy as np
import time

# using imread('path') and 1 denotes read as  color image
image01 = cv2.imread('01.jpg', 0)
image02 = cv2.imread('02.jpg', 0)
image03 = cv2.imread('03.jpg', 0)
image04 = cv2.imread('04.jpg', 0)

images = [image01, image02, image03, image04]


def show_images(images):
    for image in images:
        cv2.imshow("", image)
        cv2.waitKey()


show_images(images)


def show_filters(number, item):
    x = item.shape[0]
    y = item.shape[1]

    margins = [0, 0, 0, 0]

    # 0 for left , 1 for right , 2 for top and 3 for bottom
    for margin in range(4):
        if margin < 2:
            plc = np.squeeze(item[int(x / 2):int((x / 2) + 1), :])
        else:
            plc = np.squeeze(np.transpose(item[:, int(y / 2):int((y / 2) + 1)]))

        if margin % 2 != 0:
            plc = np.flip(plc)

        for pixel in plc:
            if pixel > 245:
                margins[margin] += 1
            else:
                break

    new_image = item[int(margins[2] + 2): int(item.shape[0] - margins[3] - 2),
                int(margins[0] + 2): int(item.shape[1] - margins[1] - 2)]

    new_x = new_image.shape[0]
    new_y = new_image.shape[1]

    cropped_x = new_x / 3
    cropped_y = new_y

    # cv2.imshow("", new_image)
    # cv2.waitKey()

    image1 = new_image[0: int(cropped_x), 0: int(cropped_y)]
    image2 = new_image[int(cropped_x): 2 * int(cropped_x), 0: int(cropped_y)]
    image3 = new_image[2 * int(cropped_x): 3 * int(cropped_x), 0: int(cropped_y)]

    # cv2.imshow("", image1)
    # cv2.waitKey()
    # cv2.imshow("", image2)
    # cv2.waitKey()
    # cv2.imshow("", image3)
    # cv2.waitKey()

    return image1, image2, image3


seperated_images = []
for i, image in enumerate(images):
    seperated_images.append(show_filters(i, image))

colorful_image = []


def show_colorful_image(i, image):
    b = image[0]
    g = image[1]
    r = image[2]
    img = cv2.merge((b, g, r))
    cv2.imshow(str(i), img)
    cv2.waitKey()

    # img = cv2.merge((g, b, r))
    # cv2.imshow("gbr", img)
    # cv2.waitKey()
    #
    # img = cv2.merge((g, r, b))
    # cv2.imshow("grb", img)
    # cv2.waitKey()
    #
    # img = cv2.merge((b, r, g))
    # cv2.imshow("brg", img)
    # cv2.waitKey()
    #
    # img = cv2.merge((r, g, b))
    # cv2.imshow("rgb", img)
    # cv2.waitKey()
    #
    # img = cv2.merge((r, b, g))
    # cv2.imshow("rbg", img)
    # cv2.waitKey()
    return img


for i, image in enumerate(seperated_images):
    colorful_image.append(show_colorful_image(i, image))


def align(img1, img2, method, off_x=(-15, 15), off_y=(-15, 15)):
    best_score = -1
    best_shift = [0, 0]

    # loop over all the different displacement permutations
    for i in range(off_x[0], off_x[1] + 1):
        for j in range(off_y[0], off_y[1] + 1):
            temp_score = score(np.roll(img1, (i, j), (0, 1)), img2, method)
            if temp_score > best_score:
                best_score = temp_score
                best_shift = [i, j]

    # return the best displaced image along with the displacement vector
    return np.roll(img1, best_shift, (0, 1)), np.array(best_shift)


def score(im1, im2, method):
    if method == 'SSD':
        return -np.sum(np.sum((im1 - im2) ** 2))
    elif method == 'NCC':
        im1 = np.ndarray.flatten(im1)
        im2 = np.ndarray.flatten(im2)
        return np.dot(im1 / np.linalg.norm(im1), im2 / np.linalg.norm(im2))


aligend_images = []
methods = ["SSD", "NCC"]


def align_images():
    for j in range(2):
        aligend = []
        for i, item in enumerate(colorful_image):
            start_time = time.time()
            ag, g_shift = align(item[:, :, 1], item[:, :, 0], methods[j])
            ar, r_shift = align(item[:, :, 2], item[:, :, 0], methods[j])

            img = cv2.merge((item[:, :, 0], ag, ar))
            end_time = time.time() - start_time
            aligend.append([str(i), img, end_time])

            # Filename
            # filename = 'image'+str(i)+'method'+methods[j]+'.jpg'
            #
            # # Using cv2.imwrite() method
            # # Saving the image
            # cv2.imwrite(filename, img)

            # cv2.imshow(str(i), img)
            # cv2.waitKey()
        aligend_images.append([methods[j], aligend])


# align_images()


def pyramid(im1, im2, method, off_x=(-4, 4), off_y=(-4, 4), depth=5):
    if im1.shape[0] < 400 or depth == 0:
        return align(im1, im2, method)
    else:
        _, best_shift = pyramid(
            cv2.resize(im1, dsize=(int((im1.shape[0]) / 2), int((im1.shape[1]) / 2)), interpolation=cv2.INTER_CUBIC),
            cv2.resize(im1, dsize=(int((im2.shape[0]) / 2), int((im2.shape[1]) / 2)), interpolation=cv2.INTER_CUBIC),
            method, depth=depth - 1)
        best_shift *= 2
        result, new_shift = align(np.roll(im1, best_shift, (0, 1)), im2, method, off_x, off_y)
        best_shift += new_shift
        return result, best_shift


def pyramid_aling():
    for j in range(2):
        aligend = []
        for i, item in enumerate(colorful_image):
            start_time = time.time()
            ag, g_shift = pyramid(item[:, :, 1], item[:, :, 0], methods[j])
            ar, r_shift = pyramid(item[:, :, 2], item[:, :, 0], methods[j])

            img = cv2.merge((item[:, :, 0], ag, ar))
            end_time = time.time() - start_time
            aligend.append([str(i), img, end_time])
            print([methods[j], str(i), end_time])

            # # Filename
            # filename = 'image pyramid '+str(i)+'method'+methods[j]+'.jpg'
            #
            # # Using cv2.imwrite() method
            # # Saving the image
            # cv2.imwrite(filename, img)
            #
            # cv2.imshow(str(i)+str(j), img)
            # cv2.waitKey()
        aligend_images.append([methods[j], aligend])


pyramid_aling()
