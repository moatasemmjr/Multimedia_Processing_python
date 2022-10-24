from unittest import result
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imsave
from PIL import Image
import cv2


def main():
    image_path_1 = r"imgs\opencv-logo.png"
    image_path_2 = r"imgs\wallpaper.jpg"
    img1 = cv2.imread(image_path_1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread(image_path_2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    _, img1_b = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY)
    img1_b = cv2.resize(img1_b, (80, 100))
    img1_b_inv = cv2.bitwise_not(img1_b)

    roi = img2[0:100, 0:80, :]
    roi_updated = cv2.bitwise_and(roi, roi, mask=img1_b_inv[:,:, 0])
    img_b_updated = cv2.bitwise_and(img1_b, img1_b, mask=img1_b[:,:, 0])
    img2[0:100, 0:80, :] = img1_b
    roi_updated = cv2.add(roi_updated, img_b_updated)
    img2[0:100, 0:80, :] = roi_updated

    plt.imshow(img2)
    plt.show()

    print()


if __name__ == '__main__':
    main()