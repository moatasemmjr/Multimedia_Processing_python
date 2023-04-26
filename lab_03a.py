from cmath import log10
from math import log2
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imsave
from PIL import Image
import cv2

def main():
    
    dft_img = cv2.imread("labs\labs\imgs\einstein_orig.tif")

    # s = c*log(1+r)
    # c = 255/log(1+M)
    M = np.max(dft_img)
    c = 255/log10(1+M)
    s = np.uint8(c*np.log10(1+dft_img))
    # s = cr^g
    s2 = (0.25)*np.power(dft_img, 2)

    plt.subplot(1,2,1)
    plt.imshow(dft_img)
    plt.subplot(1,2,2)
    plt.imshow(s)
    plt.show()

if __name__ == '__main__':
    main()