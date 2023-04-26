import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imsave
from PIL import Image
import cv2

def main():
    img = cv2.imread(r"labs\labs\imgs\wallpaper.jpg")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    b, g, r = cv2.split(img)
    colors = ['b', 'g', 'r']
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1-opencv
    hist = cv2.calcHist([img_gray], [0], None, [256], [0,256])

    # 2-numpy
    # hist = np.histogram(img_gray, 256, (0,255))[0]

    for i in range(len(colors)):
        hist_color = np.histogram(img[:,:,i], 256, (0,255))[0]
        plt.subplot(2, 5, 8+i)
        plt.plot(range(0, 256), hist_color, c=colors[i])
        plt.title(f"Hist {colors[i]}" )
    
    plt.subplot(2, 5, 6)
    for i in range(len(colors)):
        hist_color = np.histogram(img[:,:,i], 256, (0,255))[0]
        plt.plot(range(0, 256), hist_color, c=colors[i])
    plt.title(f"Hist original" )

    plt.subplot(2, 5, 1)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title("Original")
    plt.subplot(2, 5, 2)
    plt.imshow(img_gray, cmap='gray')
    plt.axis('off')
    plt.title("Gray")
    
    # 3- plt.hist
    plt.subplot(2, 5, 7)
    # plt.hist(img_gray.flatten(), 256, (0, 255), histtype='step')
    plt.plot(range(0, 256), hist)
    plt.title("Hist Gray")

# 
    plt.subplot(2, 5, 3)
    plt.imshow(b, cmap='gray')
    plt.axis('off')
    plt.title("B")
    plt.subplot(2, 5, 4)
    plt.imshow(g, cmap='gray')
    plt.axis('off')
    plt.title("G")
    plt.subplot(2, 5, 5)
    plt.imshow(r, cmap='gray')
    plt.axis('off')
    plt.title("R")

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
#     img_g = cv2.imread(r"labs\labs\imgs\washed_out_pollen_image_1.tif", 
#     cv2.IMREAD_GRAYSCALE)

#     # numpy
#     hist_np = np.histogram(img_g, 256, (0, 256))[0]
#     # hist_np = hist_np/sum(hist_np)
#     # print(hist_np)

#     # opencv
#     hist_cv2 = cv2.calcHist([img_g], [0], None, [256], [0, 256])
#     # hist_cv2 = cv2.normalize(hist_cv2,hist_cv2,0, 1, cv2.NORM_MINMAX)

#     # img_eq = np.zeros(img_g.shape)
#     img_eq = cv2.equalizeHist(img_g)
#     # clahe = cv2.createCLAHE(40, (8,8))
#     # img_eq = clahe.apply(img_g)
#     hist_eq_np = np.histogram(img_eq, 256, (0, 256))[0]

#     plt.subplot(2, 2, 1)
#     plt.imshow(img_g, cmap='gray')
#     plt.axis('off')
#     plt.title("Original Image")
#     plt.subplot(2, 2, 2)
#     plt.plot(range(0,256), hist_np)
#     # plt.hist(img_g,256, (0, 255))
#     plt.title("hist")
#     plt.subplot(2, 2, 3)
#     plt.imshow(img_eq, cmap='gray')
#     plt.axis('off')
#     plt.title("Equalized Image")
#     plt.subplot(2, 2, 4)
#     plt.plot(range(0,256), hist_eq_np)
#     plt.title("hist_equalized")
#     plt.show()

# if __name__ == '__main__':
#     main()