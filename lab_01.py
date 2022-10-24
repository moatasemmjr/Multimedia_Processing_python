import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imsave
from PIL import Image
import cv2

# pip install <package name>
# packages: matplotlib, pillow, numpy, scikit-image, opencv-python, .....

def main():
    image_path = r"imgs\wallpaper.jpg"

    # load image using PIL
    pil_image = Image.open(image_path)
    pil_image.show()
    # convert pil image to np array
    pil_to_array = np.asarray(pil_image)
    plt.imshow(pil_to_array)
    plt.axis("off")
    plt.show()
    # save the image
    array_to_pil = Image.fromarray(pil_to_array)
    array_to_pil.save(r"processed\wallpaper_new_1.jpg")

    # create random image
    random_image = np.random.randint(0, 255, (600, 800, 3), np.uint8)
    random_pil = Image.fromarray(random_image)
    random_pil.show()
    random_pil.save(r"processed\random_2.jpg")

    # # load image using skimage
    sk_image = imread(image_path)
    imshow(sk_image)
    plt.show()
    imsave(r"processed\image1_new_2.jpg", sk_image)

    # load image using opencv
    cv_image = cv2.imread(r"imgs\wallpaper.jpg")
    # Note that opencv load the image by default with BGR format, Hence if you need to deal with it using RGB format, you have to convert it.
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    imsave(r"processed\wallpaper_cv_1.jpg", cv_image)
    # to save image using openCV, it should be in BGR format
    cv2.imwrite(r"processed\wallpaper_cv_2.jpg", cv_image)
    print()


if __name__ == '__main__':
    main()