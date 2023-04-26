import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imsave
from PIL import Image
import cv2

def main():
    img = cv2.imread(r"labs\labs\imgs\Capture.PNG")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    b, g, r = cv2.split(img)
    colors = ['b', 'g', 'r']
    c = 1
    g = 3
    img = np.float32(img/255.0)
    img_power = c*img**g
    img_power = np.uint8(img_power*255)
    img_gray = cv2.cvtColor(img_power, cv2.COLOR_BGR2GRAY)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_eq1 = np.zeros(img_gray.shape, dtype=np.uint8)
    cv2.equalizeHist(img_gray, img_eq1)

    clahe = cv2.createCLAHE(10, (8,8))
    img_eq2 = clahe.apply(img_gray)

    img_eq = np.zeros(img.shape, dtype=np.uint8)
    cv2.equalizeHist(r, r)
    cv2.equalizeHist(g, g)
    cv2.equalizeHist(b, b)

    img_eq[:,:, 0] = r
    img_eq[:,:, 1] = g
    img_eq[:,:, 2] = b

    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(img_gray, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(img_eq1, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(img_eq)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
# import numpy as np
# import matplotlib

# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# from skimage.io import imread, imshow, imsave
# from PIL import Image
# import cv2

# def main():
#     img_g = cv2.imread(r"labs\labs\imgs\Capture.PNG")

#     plt.subplot(1, 2, 1)
#     plt.imshow(img_g)
#     plt.axis('off')
#     plt.title("Original Image")

#     img_g = cv2.split(img_g)
#     color = ['k', 'g', 'r']
#     plt.subplot(1, 2, 2)
#     for i, col in enumerate(color):
#         hist_np = np.histogram(img_g[i], 256, (0, 256))[0]
#         plt.plot(range(0,256), hist_np, c=col)
#     plt.title("hist")
#     plt.show()

# if __name__ == '__main__':
#     main()