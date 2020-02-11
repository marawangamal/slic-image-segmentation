import numpy as np
from kmeans import *
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':

    km = KMeans()
    img = cv2.imread("004964.jpg")
    mus, sets = km.run(img, k=10)

    plt.imshow(sets[:,:,0]); plt.show()
    import pdb; pdb.set_trace()
