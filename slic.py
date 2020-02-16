import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import timeit
from skimage import io, color

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
            mus: (int np.array) cluster centers, Shape: [K, 5]
        """

        rows, cols, D = imgvol.shape
        mus = np.zeros((k, D))

        mus_clr = np.random.randint(low=0, high=255, size=(k, 3))
        mus_rows = np.random.randint(low=0, high=rows, size=(k, 1))
        mus_cols = np.random.randint(low=0, high=cols, size=(k, 1))

        mus[:, :3] = mus_clr
        mus[:, 3:4] = mus_rows
        mus[:, 4:] = mus_cols

        return mus.astype(int)

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


        # start = timeit.default_timer()
        dists = self.compute_dists(imgvol, mus) # [rows, cols, K]
        # stop = timeit.default_timer()
        # t1 = stop - start
        # start = timeit.default_timer()
        # dists2 = self.compute_dists_vectorised(imgvol, mus) # [rows, cols, K]
        # stop = timeit.default_timer()
        # t2 = stop-start
        # print(t1, t2)


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

        self.dists = dists2

        return dists2

    def update_mus(selfs, imgvol, sets, K=2):
        """ Updates the (centers) mus given a new set assignment.
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

        return new_mus.astype(int)

    def get_color_mask(self, mus, sets, include_boundaries=False):
        """ Gets color mask, every pixel is given the color of it's assigned cluster center
        Args:
            mus: (np.array) cluster centers, Shape: [K, 5]
            sets: (np.array) integer mask for set assignments, Shape: [rows, cols, 1]
        Returns:
            clr_mask: (np.array) color mask, Shape: rows, cols, 3]
        """
        rows, cols, _ = sets.shape
        K, _ = mus.shape

        clr_mask = np.zeros((rows, cols, 3))
        broad_sets = np.zeros((rows, cols, 3))
        broad_sets[..., :] = np.copy(sets)

        # Assign pixels colours of their cluster center

        for i in range(len(mus)):
            lab = np.expand_dims(np.expand_dims(np.copy(mus[i, :3]), 0), 0)
            logicMat = broad_sets == i
            clr_mask =  clr_mask +  logicMat * lab  #[r, c, 3] [1, 1, 3]

        rgb_clr_mask = color.lab2rgb(clr_mask)
        clr_mask_with_boundaries = rgb_clr_mask


        # Add boundaries

        if(include_boundaries):

            gray_clr_mask = self.grayscale((rgb_clr_mask *255.0).astype(np.uint8))
            grad = self.gradxy(gray_clr_mask)
            grad[grad>0] = -1
            grad[grad == 0] = 1
            grad[grad == -1] = 0
            broad_grad = np.zeros((rows, cols, 3))
            broad_grad[..., :] = np.expand_dims(grad, -1)
            clr_mask_with_boundaries = rgb_clr_mask * broad_grad

        # Add circle centers

        for k in range(K):
            y, x = mus[k, 3:]
            clr_mask_with_boundaries = cv2.circle(clr_mask_with_boundaries, center=(x, y), radius=1, color=(0,0,0) , thickness=2)


        return clr_mask_with_boundaries

    def res_error(self, mus1, mus2):
        """ Calculates mean residual error, i.e. E = ||mu1 - mu2|| / |mu1|
        Args:
            mus1: (np.array) previous cluster centers, Shape: [K, D=5]
            mus2: (np.array) new cluster centers, Shape: [K, D=5]
        Returns:
            error: (float/int)
        """
        error = np.sum(np.linalg.norm(mus1 - mus2, axis=-1))

        return error/len(mus1)

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

        # Add xy coords and convert to float32
        imgvol = self.prepare_img(img)

        # Initialize
        mus = self.init_centers(imgvol, k)

        # Run till stop condition
        for i in range(iters):

            sets = self.construct_sets(imgvol, mus)
            mus = self.update_mus(imgvol, sets, k)

            self.plot(imgvol[..., :3], mus, sets)


        return mus, sets

class SLIC(KMeans):

    def init_centers_old(self, imgvol, K):
        """ SLIC Center initialization
        Args:
            imgvol: (np.array) image volume, consisting of CIElab and xy, Shape:[rows, cols, D=5]
            K: (int) number of clusters
        Return:
            mus: (np.array) cluster centers, Shape: [K, 5]
        """

        rows, cols, D = imgvol.shape
        N = rows*cols                       # Number of pixels
        S = np.sqrt(rows*cols/K)            # Stride
        mus = np.zeros((K, D))


        k=0
        # Sampling from flattened list [i, .., i+S] for the center of a mu
        for i in range(0, int((rows/S)*cols) - int(S), int(S)):

            # grid_idx = np.random.choice(np.arange(i, i+S))
            grid_idx = np.random.choice(np.arange(i+ int(S/3), i+ int(2*S/3) ))   #tighter bound
            grid_coords = self.idx2coord(grid_idx, cols)
            dist_to_center = min(S/2, (rows - grid_coords[0]*S) /2.0)
            center_coords = np.multiply(grid_coords, np.array([S, 1])) +  np.array([dist_to_center, 0])
            center_coords = center_coords.astype(int)


            # Shift center to lowest local gradient magnitude position

            if((center_coords > 3).all() and (center_coords < np.array([rows-3, cols- 3])).all() ):
                img_gray = self.grayscale(np.copy(imgvol[..., :3]))
                img_grad_mag = self.gradxy(img_gray)
                offset = np.array([center_coords[0]-3, center_coords[1]-3])
                window = img_grad_mag[center_coords[0]-3:center_coords[0]+4, center_coords[1]-3:center_coords[1]+4]
                window_idx = np.argmin(window)
                mus[k,3:]= offset + self.idx2coord(window_idx, 3)

            else:
                mus[k, 3:] = center_coords

            if (mus[k, 3]==375): import pdb; pdb.set_trace()
            k +=1

        # Populate CIElab values
        for j in range(K):

            mus[j, :3] = imgvol[int(mus[j, 3]), int(mus[j, 4]), :3]


        return mus

    def init_centers(self, imgvol, K):
        """ Center initialization at equally spaced intervals S. (S = sqrt(N/S)). Doesn't check gradients as in paper
        Args:
            imgvol: (np.array) image volume, consisting of CIElab and xy, Shape:[rows, cols, D=5]
            K: (int) number of clusters
        Returns:
            mus: (np.array) cluster centers, Shape: [K, 5]
        """
        img_h, img_w, D = imgvol.shape
        S = int(np.sqrt((img_h*img_w)/K))
        mus = np.zeros((K, D))

        # Initial positions

        h = S//2
        w = S//2
        i = 0

        while h < img_h:
            while (w < img_w and i<K):
                mus[i, :] = imgvol[h, w]
                i+=1
                w += S
            w = S // 2
            h += S
        return mus

    def compute_dists(self, imgvol, mus, m=20):
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
        S = np.sqrt(rows*cols/K) # Stride


        mus, imgvol = np.copy(mus).astype(np.float32), np.copy(imgvol).astype(np.float32)

        # Normalizing Data - Not neccessary since dividing by S and m later

        # imgvol[..., :3] = imgvol[..., :3]/ 255.0
        # mus[:,:3] = mus[:,:3]/255.0
        # imgvol[..., 3:] = np.multiply(imgvol[..., 3:], 1/np.array([rows, cols]))
        # mus[:,3:] = np.multiply(mus[:,3:], 1/np.array([rows, cols]))


        img_lab, img_yx= imgvol[..., :3], imgvol[..., 3:]    # [rows, cols, 3], [rows, cols, 2]


        dists_yx = np.ones((rows, cols, K, 2)) * np.inf
        dists_lab = np.ones((rows, cols, K, 3)) * np.inf

        mu_lab_broadcast = np.zeros((1, 1, 1, 3))
        mu_yx_broadcast = np.zeros((1, 1, 1,2))

        for j in range(K):
            mu_lab_broadcast[..., :] = mus[j, :3]
            mu_yx_broadcast[..., :] = mus[j, 3:]

            # center_r, center_c = np.multiply(mus[j, 3:], np.array([rows, cols])) # if Normalizing
            center_r, center_c = mus[j, 3:]

            r1 = max(0, center_r  -  S)
            r2 = min(center_r + S+1, rows)
            c1 = max(0, center_c - S)
            c2 = min(center_c + S+1, cols)

            r1, r2, c1, c2 = int(r1), int(r2), int(c1), int(c2)

            dists_lab[r1:r2, c1:c2, j, :] = np.abs(img_lab[r1:r2, c1:c2] - mu_lab_broadcast ) #[rows, cols, 1, 3] - [1,1,1,3]
            dists_yx[r1:r2, c1:c2, j, :] = np.abs(img_yx[r1:r2, c1:c2] - mu_yx_broadcast)  #[rows, cols, 1, 2] - [1,1,1,2]

        dists_lab = np.linalg.norm(dists_lab, axis=-1)  # [rows, cols, K]
        dists_yx = np.linalg.norm(dists_yx,  axis=-1)   # [rows, cols, K]


        dists = np.zeros((rows, cols, K, 2))

        dists[..., 0] = dists_lab/m
        dists[..., 1] = dists_yx/S
        dists = np.linalg.norm(dists,  axis=-1)

        self.dists = dists

        return dists

    def compute_dists_vectorised(self, imgvol, mus, m=20):
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
        S = np.sqrt(rows*cols/K) # Stride


        MUS_lab = np.zeros((1,1,K, 3))
        MUS_yx = np.zeros((1,1,K, 2))
        IMG_lab, IMG_yx = np.zeros((rows, cols, 1, 3)), np.zeros((rows, cols, 1, 2))

        IMG_lab[:,:, 0, :], IMG_yx[:,:, 0, :]= np.copy(imgvol[..., :3]), np.copy(imgvol[..., 3:])
        MUS_lab[0,0,:,:] = mus[:,:3]
        MUS_yx[0,0,:,:] = mus[:,3:]


        # Broadcasting: IMG_lab[rows, cols, 1, 3] - musk[1,1,K,3] = [rows, cols, K, 3]
        dists_lab = np.linalg.norm( IMG_lab - MUS_lab , axis = -1)      # [rows, cols, K]
        dists_yx = np.abs(IMG_yx - MUS_yx)                              # [rows, cols, K, 2]
        dists_yx[dists_yx > S] = np.inf
        dists_yx = np.linalg.norm( dists_yx , axis = -1)         # [rows, cols, K]

        # dists_yx[dists_yx > S] = np.inf

        dists = np.zeros((rows, cols, K, 2))
        dists[..., 0] = dists_lab/m
        dists[..., 1] = dists_yx/S
        dists = np.linalg.norm(dists,  axis=-1)


        self.dists = dists

        return dists

    def idx2coord(self, index, cols):
        """ Converts an index to a coordinate
        Args:
            index: (int)
            cols: int
        Returns:
            coords: (np.array) 1d array containing coordiates
        """
        col = index % cols
        row = int(index/cols)
        coords = np.array([row, col])

        return coords

    """ Gradient Computation """

    def gradxy(self, img, sigma=2, smooth=False):
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
        img_new = np.copy(img)
        if(smooth):
            img_new = self.convolve(np.copy(img), self.gaus(sigma))

        # Compute gradients
        Ix = self.convolve(img_new, fx)
        Iy = self.convolve(img_new, fy)


        img_grad[..., 0], img_grad[..., 1] = Ix, Iy
        img_grad_mag = np.linalg.norm(img_grad, axis=-1)

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

    def remove_strays(self, sets):
            """ Relabels stray pixels - Not completed
            """

            import pdb; pdb.set_trace()

            rows, cols, D = sets.shape

            left_shift = np.zeros((rows, cols, D))
            right_shift = np.zeros((rows, cols, D))
            up_shift = np.zeros((rows, cols, D))
            down_shift = np.zeros((rows, cols, D))
            conn = np.zeros(rows, cols, 4)


            #  Shapes:[rows, cols, 1]
            right_shift[:, 1:, :] = sets[:, :-1, :]
            right_shift[:,0,:] = sets[:, 0, :]

            left_shift[:, :-1, :] = sets[:, 1:, :]
            left_shit[:,-1,:] = sets[:, -1, :]

            up_shift[1:, :, :] = sets[:-1, :, :]
            up_shift[0,:,:] = sets[0, :, :]

            down_shift[:-1, :, :] = sets[1:, :, :]
            left_shift[:,-1,:] = sets[:, -1, :]

            # Shapes:[rows, cols, 1]
            same_right = sets == right_shift
            same_left = sets == left_shift
            same_up = sets == up_shift
            same_down = sets == down_shift

            score = same_right + same_left + same_down + same_up
            score = score < 2

            re_labelled = score*left_shift + sets

            return re_labelled

    def run(self, img, k, iters=10):
        """ Runs SLIC. Algorithm summary:
            1. Initilize centers in grid like fashion at intervals of S
            2. Construct Sets using distance measure
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

        # Add xy coords and convert to float32
        imgvol = self.prepare_img(img)

        # Initialize
        mus = self.init_centers(imgvol, k)

        # Run till stop condition
        for i in range(iters):

            sets = self.construct_sets(imgvol, mus)
            old_mus = np.copy(mus)
            mus = self.update_mus(imgvol, sets, k)
            error = self.res_error(np.copy(old_mus), np.copy(mus))
            print("iter: {} Error: {}".format(i, error))

            if(error < 3):
                break

        return mus, sets
