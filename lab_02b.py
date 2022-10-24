from unittest import result
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imsave
from PIL import Image
import cv2


def image_addition(image_1, image_2, weight=0.5):
    result = weight*image_1 + (1-weight)*image_2
    result = np.clip(result, 00, 255)
    result = np.uint8(result)
    return result

def main():
    image_path_1 = r"imgs\opencv-logo.png"
    image_path_2 = r"imgs\wallpaper.jpg"
    img1 = cv2.imread(image_path_1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread(image_path_2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    h, w = img1.shape[:2]
    img2 = cv2.resize(img2, (w, h))

    img3 = image_addition(img1, img2, 0.2)
    img4 = cv2.addWeighted(img1, 0.2, img2, 0.8, 0)
    

    plt.imshow(img4)
    plt.show()

    print()


if __name__ == '__main__':
    main()