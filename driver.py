import numpy as np
from kmeans import *
import cv2
import matplotlib.pyplot as plt
from skimage import io, color

if __name__ == '__main__':

    # IMPORT IMAGE

    img = cv2.imread("2.jpg")
    img_CIElab = cv2.cvtColor(np.copy(img), cv2.COLOR_BGR2LAB)


    # KMEANS EXAMPLE

    # km = KMeans()
    # mus, sets = km.run(img, k=10)
    # plt.imshow(sets[:,:,0]); plt.show()


    # SLIC EXAMPLE

    rgb2 = io.imread("2.jpg",plugin='matplotlib')
    img_CIElab2 = color.rgb2lab(rgb2)

    sl = SLIC()
    mus, sets, = sl.get_superpixels(img_CIElab2, k=100, iters=5)
    clr_mask = sl.get_color_mask(mus, sets)
    plt.imshow(clr_mask); plt.show()
