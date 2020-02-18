import numpy as np
from slic import *
import cv2
import matplotlib.pyplot as plt
from skimage import io, color

from max_sum import *

if __name__ == '__main__':

    # IMPORT IMAGE
    img = io.imread("imgs/0001.jpg",plugin='matplotlib')
    img_CIElab = color.rgb2lab(img)


    # KMEANS EXAMPLE

    # km = KMeans()
    # mus, sets = km.run(img, k=10)
    # plt.imshow(sets[:,:,0]); plt.show()


    # SLIC EXAMPLE
    sl = SLIC()
    mus, sets, = sl.run(img_CIElab, k=100, iters=1)
    clr_mask = sl.get_color_mask(mus, sets)

    # Generate random unary pottentials
    rows, cols, _ = sets.shape
    unarys = np.ones((rows, cols, 5))
    unarys[:,:int(cols/2), 1] = 5
    unarys[:,int(cols/2):, 3] = 5
    unarys = unarys/6

    # Build graph
    graph = graph_builder(sets, mus[:96], unarys)
    nodes = graph.get_nodes()

    # Execute Loopy Belief Propagation on graph
    lbp = LBP(nodes, 5)
    energys, labels = lbp.run()
