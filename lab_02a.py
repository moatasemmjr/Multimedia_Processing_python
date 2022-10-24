from email.mime import image
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imsave
from PIL import Image
import cv2

def image_comp(input_image:np.ndarray) ->np.ndarray:
    return 255-input_image


def main():
    image_path = r"imgs\opencv-logo.png"
    logo_np = cv2.imread(image_path)
    logo_np = cv2.cvtColor(logo_np, cv2.COLOR_BGR2RGB)
    logo_np_resized = cv2.resize(logo_np, (80, 100), cv2.INTER_LANCZOS4)
    logo_np_comp1 = image_comp(logo_np_resized)
    logo_np_comp2 = cv2.bitwise_(logo_np_resized)

    plt.imshow(logo_np_comp2)
    plt.show()
    print()


if __name__ == '__main__':
    main()