import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imsave
from PIL import Image
import cv2

def stretch(img, r1, s1, r2, s2):
    img_copy = img.copy()
    for i in range(len(img)):
        for j in range(len(img[i])):
            pix = img[i, j]
            if(pix>=0 and pix<=r1):
                pix = (s1/r1) * pix
            elif (pix> r1 and pix <= r2):
                pix = ((s2-s1)/(r2-r1)) * (pix-r1) + s1
            else:
                pix = ((255-s2)/(255-r2)) * (pix-r2) + s2
            img_copy[i,j] = pix
    return np.uint8(img_copy)

def main():
    
    # img = cv2.imread(r"labs\labs\imgs\einstein_orig.tif", cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(r"labs\labs\imgs\Capture.PNG" , cv2.IMREAD_GRAYSCALE )
    height, width = img.shape[:2]

    # log
    # s = c*log(1+r)
    # c = (L-1)/log(1+M), L-1(8-bit) = 255
    # M = max value in image
    M = np.max(img)
    c = 255/np.log10(1+M)
    img_log = np.uint8(c * np.log10(1+img))
    # log inverse 
    # r = exp(s\c)-1
    # img2 = np.exp(img_log/c)-1

    # power
    # s = c*r^g
    c = 1
    g = 2 # 
    img = np.float32(img/255.0)  
    img_power = c*img**g
    img_power = np.uint8(img_power*255)

    # strtech
    img = np.uint8(img*255)
    img_stretch = stretch(img, 70, 40, 140, 200)

    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray') 
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(img_stretch, cmap='gray')
    plt.axis("off")
    plt.show()

if __name__ == '__main__':
    main()
