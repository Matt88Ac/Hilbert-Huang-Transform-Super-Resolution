from PIL.Image import fromarray
import cv2
import numpy as np
from scipy import ndimage, interpolate
from skimage.morphology import local_maxima, local_minima


class EMD2D:
    """ EMD2D implements an EMD decomposition for a 2-D signal such as images.
    """
    imfs: np.ndarray = np.array([])
    NoIMFS: int = 0

    # TODO: Add functionality to decide for the stopping critertion.
    def __init__(self, img: np.ndarray, S_critertion=0, max_IMFs=10):
        self.MAX = 1000
        self.S_critertion = S_critertion
        self.max_IMFs = max_IMFs
        self.imfs = self.EMD(img)

    @classmethod
    def find_local_extrema(cls, img):
        """ Class method to find local extrema in a given image.
            The method returns indices of maximum and minimum respectively.
            Each represented by a tuple of array for each dimension.
        """
        max_points = local_maxima(img, indices=True)
        min_points = local_minima(img, indices=True)

        return max_points, min_points

    @classmethod
    def envelope(cls, img):
        """
        Class method returns splines created out of local maximum
        and minimum points in the image.
        """
        max_points, min_points = cls.find_local_extrema(img)

        px_max = max_points[0]
        py_max = max_points[1]

        px_min = min_points[0]
        py_min = min_points[1]

        max_values = img[local_maxima(img)]
        min_values = img[local_minima(img)]

        def spline(X, Y, Z):
            return interpolate.interp2d(X, Y, Z, kind='cubic')

        splineMax = spline(px_max, py_max, max_values)
        splineMin = spline(px_min, py_min, min_values)

        newx = np.arange(0, img.shape[0], 0.01)
        newy = np.arange(0, img.shape[1], 0.01)

        newx, newy = np.meshgrid(newx, newy)

        return splineMax(newx, newy), splineMin(newx, newy)

    @classmethod
    def count_zero_crossings(cls, img):
        """ Class method returns number of zero crossings in given image.
            https://homepages.inf.ed.ac.uk/rbf/HIPR2/zeros.htm
        """
        LoG = ndimage.gaussian_filter(img, 2)
        thres = np.absolute(LoG).mean() * 0.75
        rows = img.shape[0]
        columns = img.shape[1]

        def return_neighbors(im, n, k):
            return np.array([im[n - 1, k], img[n + 1, k], img[n, k - 1], img[n, k + 1]])

        output = np.zeros(LoG.shape)
        for i in range(1, rows - 1):
            for j in range(1, columns - 1):
                neighbors = return_neighbors(LoG, i, j)
                p = LoG[i, j]
                max_p = neighbors.max()
                min_p = neighbors.min()
                if p > 0:
                    zero_cross = True if min_p < 0 else False
                else:
                    zero_cross = False if max_p > 0 else False
                if (max_p - min_p) > thres and zero_cross:
                    output[i, j] = 1

        return np.count_nonzero(output == 1)

    @classmethod
    def end_condition(cls, image, IMFs):
        rec = np.sum(IMFs, axis=0)

        # If reconstruction is perfect, no need for more tests
        if np.allclose(image, rec):
            return True

        return False

    def EMD(self, img):
        img_min, img_max = img.min(), img.max()
        offset = img_min
        scale = img_max - img_min

        img_s = (img - offset) / scale

        def sift(imfK):
            """ Apply the sifting procedure on the given 2-D signal. """
            max_envelope, min_envelope = self.envelope(imfK)
            mean = (max_envelope + min_envelope) * 0.5
            imfK = imfK - mean
            return imfK

        n = 0  # Number of IMFs

        # Creates a tensor such each matrix represents an IMF
        IMFs = np.empty((n,) + img.shape)

        while True:
            # At the k-th iteration of the decomposition, 
            # we refer to data as the original signal after being subtracted from the k-1 generated IMFs.
            x = img_s - np.sum(IMFs[:n], axis=0)

            k = 0  # Iterations for current IMFs
            k_h = 0  # number of consecutive iterations to compare to S number (explained below)
            flag = True
            while flag and k < self.MAX:
                # The code use the S number criterion, i.e the canidate will be elected as IMF after s consecutive
                # runs in which the difference between local extrema and zero crossing is by at most 1. S is
                # pre-determined.

                imf = sift(x)
                imf_old = imf.copy()
                zero_crossing = EMD2D.count_zero_crossings(imf)
                local_max, local_min = self.find_local_extrema(imf)

                diff = (len(local_max[0]) + len(local_min[0])) - zero_crossing
                if abs(diff) < 1:
                    k_h += 1
                else:
                    k_h += 0

                if k_h == self.S_critertion:
                    flag = False

                    # Add the chosen canidate to the IMFs
                IMFs = np.vstack((IMFs, imf.copy()[None, :]))
                n += 1

                if self.end_condition(img, IMFs) or (0 < self.max_imf <= n):
                    notFinished = False
                    break

            res = img_s - np.sum(IMFs[:n], axis=0)
            if not np.allclose(res, 0):
                IMFs = np.vstack((IMFs, res[None, :]))
                n += 1

            self.NoIMFS = n
            IMFs = IMFs * scale
            IMFs[-1] += offset
            return IMFs


imag = cv2.imread('DATA/input.jpg', 0)
emd = EMD2D(imag, 0.01)
