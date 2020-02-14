import numpy as np
from kmeans import *
import cv2
import matplotlib.pyplot as plt
from skimage import io, color

if __name__ == '__main__':

    # IMPORT IMAGE

    img = io.imread("imgs/bear.jpg",plugin='matplotlib')
    img_CIElab = color.rgb2lab(rgb2)


    # KMEANS EXAMPLE

    # km = KMeans()
    # mus, sets = km.run(img, k=10)
    # plt.imshow(sets[:,:,0]); plt.show()


    # SLIC EXAMPLE

    sl = SLIC()
    mus, sets, = sl.get_superpixels(img_CIElab, k=100, iters=5)
    clr_mask = sl.get_color_mask(mus, sets)
    plt.imshow(clr_mask); plt.show()