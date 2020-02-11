
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    """ Kmeans for image data.
    Steps for use:
        1. Instantiate
        2. Call run method and pass in arguments as specified
    """

    def init_centers(self, imgvol, k):
        """ Random center initialization
        Args:
            imgvol: (np.array) image volume, consisting of CIElab and xy, Shape:[rows, cols, D=5]
            k: (int) number of clusters
        Return:
            mus: (np.array) cluster centers, Shape: [K, 5]
        """

        rows, cols, D = imgvol.shape
        mus = np.zeros((k, D))

        mus_clr = np.random.randint(low=0, high=255, size=(k, 3))
        mus_rows = np.random.randint(low=0, high=rows, size=(k, 1))
        mus_cols = np.random.randint(low=0, high=cols, size=(k, 1))

        mus[:, :3] = mus_clr
        mus[:, 3:4] = mus_rows
        mus[:, 4:] = mus_cols

        return mus

    def prepare_img(self, img, include_xy=True):
        """ Reshapes image, appends xy coords to image and converts to numpy array
        Args:
            img: (np.array or cv) input image of unknown dtype. Shape: [3, rows, cols]
        Returns:
            imgvol: (np.array float32) output image vol with yx coords appended. Shape: [rows, cols, 5]
        """

        r, c, _ = img.shape
        imgvol = np.zeros((r, c, 5))

        if (include_xy):
            x = np.arange(c)
            y = np.arange(r)
        else:
            x = np.arange(c) * 0
            y = np.arange(r) * 0


        cols, rows = np.meshgrid(x, y)

        imgvol[:,:,:3] = img
        imgvol[:,:,3] = rows
        imgvol[:,:,4] = cols

        return imgvol.astype(np.float32)

    def construct_sets(self, imgvol, mus):
        """ Assign img pixels to the set with closest center mu
        Args:
            imgvol: (np.array) image volume, consisting of CIElab and xy, Shape:[rows, cols, D=5]
        Returns:
            mask: (np.array) integer mask for set assignments, Shape: [rows, cols, 1]
        """

        dists = self.compute_dists(imgvol, mus) # [rows, cols, K]
        mask = np.argmin(dists, axis = -1)

        return np.expand_dims(mask, -1)

    def compute_dists(self, imgvol, mus):
        """ Computes distances between pixels and centers.
        Args:
            imgvol: (np.array) input image, Shape: [rows, cols, D=5]
            mus: (np.array) cluster centers, Shape: [K, 5]
        Returns:
            dists: (np.array) output distances, Shape:[rows, cols, K]
        """

        rows, cols, D = imgvol.shape
        K, _ = mus.shape
        dists2 = np.zeros((K, rows, cols))

        MUS = np.zeros((1,1,K, D))
        MUS[0,0,:,:] = mus
        IMG = np.zeros((rows, cols, 1, D))
        IMG[:,:,0,:] = np.copy(imgvol)


        # Loop alternative
        # dists = np.zeros((K, rows, cols))
        # for k in range(K):
        #     # broadcasting img[rows, cols, 5] - musk[1,5]
        #     dists[k] = np.linalg.norm( np.copy(imgvol) - mus[k], axis = 2)

        # Broadcasting img[rows, cols, 1, 5] - musk[1,1,K,5] = [rows, cols, K, 5]
        dists2 = np.linalg.norm( IMG - MUS , axis = -1)


        return dists2

    def update_mus(selfs, imgvol, sets, K=2):
        """ Updates the (centers) mus given a new set assignment.        Input S and Img are (R, C) and (R,C, D)
        Args:
            imgvol: (np.array) image volume, consisting of CIElab and xy, Shape:[rows, cols, D=5]
            sets: (np.array) integer mask for set assignments, Shape: [rows, cols, 1]
            K: (int) number of clusters
        Returns:
            new_mus: (np.array) cluster centers, Shape: [K, 5]
        """

        rows, cols, D = imgvol.shape
        new_mus = np.zeros((K, D))


        for k in range(K):
            logicMat = np.ones((rows,cols,1))*k == sets
            num_points = np.sum(logicMat)

            # braodcasting: logicMat[r, c, 1] x imgvol[r, c, 5] = [r, c, 5] then summing across row and columns
            new_mus[k] = np.sum(np.sum(logicMat * imgvol, axis = 0), axis = 0)

            # Averaging done later to avoid division by zero in case no assigments to this cluster
            if (num_points > 0):
                new_mus[k] = new_mus[k]/ num_points

        return new_mus

    def run(self, img, k, iters=10):
        """ Runs K-Means.
            1. Initilize centers
            2. Construct Sets
            3. Update mus (i.e. recalculate centers based on set memberships)
            4. If converged return mus and sets, else goto (2)
        Args:
            img: (np.array) input image, Shape: [3, rows, cols]
            k: (int) number of clusters
            iters: (int) number of iterations
        Returns
            mus: (np.array) cluster centers, Shape: [K, 5]
            sets: (np.array) cluster assignments, Shape: [3, rows, cols]
        """

        # import pdb; pdb.set_trace()
        # Add xy coords and convert to float32
        imgvol = self.prepare_img(img)

        # Initialize
        mus = self.init_centers(imgvol, k)

        # Run till stop condition
        for i in range(iters):
            sets = self.construct_sets(imgvol, mus)
            mus = self.update_mus(imgvol, sets, k)

        return mus, sets


class SLIC(Kmeans):

    def compute_dists(self, imgvol, mus, m=10):
        """ Computes distances between pixels and centers, dist=infinity if the xy dist is > 2S
        Args:
            imgvol: (np.array) input image, Shape: [rows, cols, D=5]
            mus: (np.array) cluster centers, Shape: [K, 5]
            m: (float) Hyperparameter, weighting for spatial proximity vs colour
        Returns:
            dists: (np.array) output distances, Shape:[rows, cols, K]
        """

        rows, cols, D = imgvol.shape
        K, _ = mus.shape
        S = np.sqrt(rows*cols/K)

        MUS_lab = np.zeros((1,1,K, 3))
        MUS_yx = np.zeros((1,1,K, 2))
        IMG_lab, IMG_yx = np.zeros((rows, cols, 1, 3)), np.zeros((rows, cols, 1, 2))


        IMG[:,:,0,:3], IMG[:,:,0,3:] = np.copy(imgvol[..., :3]), np.copy(imgvol[..., 3:])
        MUS_lab[0,0,:,:3] = mus[:,:3]
        MUS_yx[0,0,:,3:] = mus[:,3:]


        # Broadcasting IMG_lab[rows, cols, 1, 3] - musk[1,1,K,3] = [rows, cols, K, 3]
        dists_lab = np.linalg.norm( IMG_lab - MUS_lab , axis = -1)  # [rows, cols, K]
        dists_yx = np.linalg.norm( IMG_yx - MUS_yx , axis = -1)     # [rows, cols, K]

        # dists_yx = IMG_yx - MUS_yx  # [rows, cols, K, 2]
        # logicMat = dists_yx > 2*S   # [rows, cols, K, 2]
        # logicMat = np.max(logicMat, axis=-1)

        dists_yx[dists_yx > 2*S] = np.inf
        dists = dists_lab + (m/S)*dists_yx

        return dists

    def init_centers(self, imgvol, K, S):
        """ SLIC Center initialization
        Args:
            imgvol: (np.array) image volume, consisting of CIElab and xy, Shape:[rows, cols, D=5]
            k: (int) number of clusters
        Return:
            mus: (np.array) cluster centers, Shape: [K, 5]
        """
        rows, cols, D = imgvols.shape
        N = rows*cols #number of pixels
        mus = np.zeros((K, D))

        k=0


        # Sampling from flattened list [i, .., i+S] for the center of a mu
        for i in range(0,N-S, S):

            center_idx = np.random.choice(np.arrange(i, i+S))
            center_coords = self.idx2coord(center_idx, rows, cols)

            # Shift center to lowest local gradient magnitude position
            if((center_coord > 3).all() and (center_coord < np.array([rows-3, cols, 3])).all() ):
                img_gray = self.grayscale(np.copy(img))
                img_grad_mag = self.gradxy(img_gray)
                offset = np.array([center_coords[0]-3, center_coords[1]-3])
                window = img_grad_mag[center_coords[0]-3:center_coords[0]+4, center_coords[1]-3:center_coords[1]+4]
                window_idx = np.argmin(window)
                mus[k,3:]= offset + self.idx2coord(window_idx, 3, 3)

            k +=1

        # Populate CIElab values
        for j in range(K):
            mus[j, :3] = img[mus[3], mus[4], :3]


        return mus

    def idx2coord(self, index, rows, cols):
        """ Converts an index to a coordinate
        Args:
            index: (int)
            rows: (int)
            cols: int
        Returns:
            coords: (np.array) 1d array containing coordiates
        """
        col = index % width
        row = int(index/width)
        coords = np.array([row, col])

        return coords

    """ Gradient Computation """

    def gradxy(self, img, sigma=2):
        """ Image Gradient computation through convolution. Image smoothed before gradient calculation
        Args:
            img: (np array) Grayscale input image, Shape [Rows, Cols]
            sigma: (int) std deviation for gaussian filter.
        Returns:
            img_grad_mag: (np array) gradient magnitude of image, Shape: [Rows, Cols]
        """

        rows, cols = img.shape
        img_grad = np.zeros((rows, cols, 2))

        # Filters
        fx = np.array([[1,-1]])
        fy = np.array([[1],[-1]])

        # Smooth before gradient calculation
        img_new = self.convolve(np.copy(img), self.gaus(sigma))

        # Compute gradients
        Ix = self.convolve(img_new, fx)
        Iy = self.convolve(img_new, fy)


        img_grad[..., 0], img_grad[..., 1] = Ix, Iy
        img_grad_mag = np.linalg.norm(img_grad, -1)

        return img_grad_mag

    def convolve(self, img, filter):
        """ Convolution operation
        Args:
            img: (np.array) input image, Shape [rows, cols]
        Returns:
            convolved image of same size
        """

        return ndimage.convolve(img,  filter, mode='constant')

    def gaus(self, sigma):
        """ Creates a gaussian kernel for convolution
        Args:
            sigma: (float) std deviation of gaussian
        Returns:
            gFilter: (np.array) filer, Shape: [5,5]
        """
        kSize = 5
        gFilter = np.zeros((kSize, kSize))
        gausFunc = lambda u,v, sigma : (1/(2*np.pi*(sigma**2))) * np.exp( - ( (u**2) + (v**2) )/ (2 * (sigma**2)) )

        centerPoint = kSize//2

        for i in range(kSize):
            for j in range(kSize):

                u = i - centerPoint
                v = j - centerPoint

                gFilter[i, j] = gausFunc(u,v,sigma)


        return gFilter

    def grayscale(self, img):
        """ RGB to Grayscale conversion
        Args:
            img: (np array) input image, Shape [3, Rows, Cols]
        Returns:
            img_gray: (np array) input image, Shape [Rows, Cols]
        """

        img_gray = img

        # Omit if already grayscale
        if(len(img.shape) > 2):
            img_gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img_gray

    def get_superpixels(self, img, K):
        """ SLIC.
        Args:
            K: (int) number of cluster centers/super-pixels
        """
        rows, cols, D = img.shape
        N = rows*cols # number of pixels
        S = np.sqrt(N, K) # Step size - will initilize centers to be at equally spaced steps in grid
        img = img.astype(np.float32)
