from PIL.Image import fromarray
import cv2
import numpy as np
from scipy import ndimage, interpolate, signal
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
    def find_local_extrema(cls, img: np.ndarray):
        """ Class method to find local extrema in a given image.
            The method returns indices of maximum and minimum respectively.
            Each represented by a tuple of array for each dimension.
        """
        max_points = local_maxima(img, indices=True)
        min_points = local_minima(img, indices=True)

        return max_points, min_points

    @classmethod
    def envelope(cls, img: np.ndarray):
        """
        Class method returns splines created out of local maximum
        and minimum points in the image.
        """

        max_points, min_points = cls.find_local_extrema(img)

        def spline(X, Y, Z):
            spl = interpolate.SmoothBivariateSpline(X, Y, Z)
            # spl = interpolate.interp2d(X, Y, Z, kind='cubic')
            # spl = interpolate.Rbf(X, Y, Z)
            return spl

        # footprint = ndimage.generate_binary_structure(2, 2)

        # minPoints = ndimage.minimum_filter(img, footprint=footprint) == img
        # maxPoints = ndimage.maximum_filter(img, footprint=footprint) == img

        # background = ndimage.binary_erosion(img == 0, structure=footprint, border_value=1)

        # minPoints = minPoints ^ background
        # maxPoints = maxPoints ^ background

        splineMax = spline(max_points[0], max_points[1], img[local_maxima(img)])
        splineMin = spline(min_points[0], min_points[1], img[local_minima(img)])

        nx = np.arange(0, img.shape[0])
        ny = np.arange(0, img.shape[1])

        # newx, newy = np.meshgrid(nx, ny)

        mx = splineMax(nx, ny).astype(float)
        mn = splineMin(nx, ny).astype(float)
        return mx, mn  # np.nonzero(maxPoints), np.nonzero(minPoints)

    @classmethod
    def count_zero_crossings(cls, img: np.ndarray):
        """ Class method returns number of zero crossings in given image.
            https://homepages.inf.ed.ac.uk/rbf/HIPR2/zeros.htm
        """
        LoG = ndimage.gaussian_filter(img, 2)
        thres = np.absolute(LoG).mean() * 0.75

        bin_struct = ndimage.generate_binary_structure(2, 2)
        mins = (ndimage.maximum_filter(-LoG, footprint=bin_struct) == -LoG).astype(int)
        maxs = (ndimage.maximum_filter(LoG, footprint=bin_struct) == LoG).astype(int)

        mins = mins * LoG
        mins = (mins < 0).astype(int)

        maxs = maxs * LoG
        maxs = (maxs > 0).astype(int)

        biggerThanZero = LoG > 0
        smallerThanZero = LoG < 0

        fp = np.ones((3, 3))

        biggerThanZero = signal.convolve2d(biggerThanZero, fp, mode='same')
        smallerThanZero = signal.convolve2d(smallerThanZero, fp, mode='same')

        biggerThanZero = (biggerThanZero > 0).astype(int)
        smallerThanZero = (smallerThanZero > 0).astype(int)

        biggerThanZero *= mins
        smallerThanZero *= maxs

        mins = (ndimage.maximum_filter(-LoG, footprint=bin_struct) == -LoG).astype(int) * LoG
        maxs = (ndimage.maximum_filter(LoG, footprint=bin_struct) == LoG).astype(int) * LoG

        output = ((maxs - mins) > thres).astype(int)

        output = output * biggerThanZero + output * smallerThanZero
        """""""""
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
        """
        return output.sum()

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
                zero_crossing = self.count_zero_crossings(imf)
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

                if self.end_condition(img, IMFs) or (0 < self.max_IMFs <= n):
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
