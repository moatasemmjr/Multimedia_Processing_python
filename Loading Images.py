from matplotlib import image
from skimage.io import imsave, imread, imshow
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    
    # load an image using skimage kit library 
    path_image = "logo.png"
    sk_image = imread(path_image)
    imsave("logo2.png", sk_image)
    imshow(sk_image)
    plt.show()
    
    # load an image using pil "matplotlib" library
    pil_image = Image.open(path_image)
    pil_image.show()
    pil_image.save("pil_image.jpg")
    # image_array = np.asarray(pil_image) # get the data array of a pil "matplotlib" library
    
    
    #creating a random image using np library + pil library
    random_pil = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    rand_pil = Image.fromarray(random_pil)
    rand_pil.show()
    rand_pil.save("random_image.png")
    
    # load an image using cv2 library
    cv_image = cv2.imread(path_image)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) #The cv2 library loads the image in BGR extension , I used this line to convert from BGR to RGB
    plt.imshow(cv_image)
    plt.show()
    cv2.imwrite("cv_image.png", cv_image)
    
    print()
    
if __name__  == '__main__':
    main()
 
# -----------------------------------------------------

# pip install scikit-image
# pip install opencv-python
# pip install matplotlib

# -----------------------------------------------------