import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imsave
from PIL import Image
import cv2

def main():
    img2 = cv2.imread(r"labs\labs\imgs\Capture.PNG",cv2.IMREAD_GRAYSCALE)
    img = img2.copy()
    height, width = img.shape[:2]

    bit_planes = []
    img_ones = np.uint8(np.ones(img.shape, dtype=np.uint8))
    for i in range(8):
        current_img = cv2.bitwise_and(img, img_ones)
        bit_planes.append(255*current_img)
        img = np.uint8(img/2)
        cv2.imwrite(rf"labs\labs\processed\{i}q.jpg", current_img*255)

    img_org = 128 * (bit_planes[7]/255.0) + 64 * (bit_planes[6]/255.0) + 32 * (bit_planes[5]/255.0)
    cv2.imwrite(r"labs\labs\processed\summ.jpg", img_org)
    plt.subplot(3,3,1)
    plt.title("Original")
    plt.imshow(img2, cmap='gray')
    plt.axis("off")

    for i in range (0, 8):
        plt.subplot(3,3,i+2)
        plt.imshow(bit_planes[i], cmap='gray')
        plt.axis("off")
        # plt.title(str(i) + "-bit plane")
        plt.title(f"{i} -bit plane")
    plt.show()

if __name__ == '__main__':
    main()