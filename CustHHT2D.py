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
    def __init__(self, S_critertion=6, max_IMFs=10):
        self.MAX = 1000
        self.S_critertion = S_critertion
        self.max_imf = max_IMFs
            
    def find_local_extrema(self, img):
        """ Class method to find local extrema in a given image.
            The method returns indices of maximum and minimum respectively.
            Each represented by a tuple of array for each dimension.
        """
        max_points = local_maxima(img, indices=True)
        min_points = local_minima(img, indices=True)

        return max_points, min_points
        
    def envelope(self, img):
        """
        Class method returns splines created out of local maximum
        and minimum points in the image.
        """
        max_points, min_points = self.find_local_extrema(img)

        """
        px_max = max_points[0]
        py_max = max_points[1]

        px_min = min_points[0]
        py_min = min_points[1]

        max_values = img[px_max, py_max]
        min_values = img[px_min, py_max]
        """

        def spline(X, Y, Z):
            return interpolate.SmoothBivariateSpline(X, Y, Z)

        """
        splineMax = spline(px_max, py_max, max_values)
        splineMin = spline(px_min, py_min, min_values)

        newx = np.arange(0, img.shape[0], 0.01)
        newy = np.arange(0, img.shape[1], 0.01)

        newx, newy = np.meshgrid(newx, newy)
        """

        splineMax = spline(max_points[0], max_points[1], img[local_maxima(img)])
        splineMin = spline(min_points[0], min_points[1], img[local_minima(img)])


        nx = np.arange(0, img.shape[0])
        ny = np.arange(0, img.shape[1])

        # newx, newy = np.meshgrid(nx, ny)

        mx = splineMax(nx, ny).astype(float)
        mn = splineMin(nx, ny).astype(float)
        return mx, mn  # np.nonzero(maxPoints), np.nonzero(minPoints)

        #return splineMax(newx, newy), splineMin(newx, newy)

    
    @classmethod
    def count_zero_crossings(cls, img):
        """ Class method returns number of zero crossings in given image.
            https://homepages.inf.ed.ac.uk/rbf/HIPR2/zeros.htm
        """
        LoG = ndimage.gaussian_filter(img, 2)
        thres = np.absolute(LoG).mean() * 0.75 
        rows = img.shape[0]
        columns = img.shape[1]

        def return_neighbors(img,i,j):
            return np.array([img[i-1,j],img[i+1,j],img[i,j-1],img[i,j+1]])

        output = np.zeros(LoG.shape)
        for i in range(1, rows-1):
            for j in range(1, columns-1):
                neighbors = return_neighbors(LoG,i,j)
                p = LoG[i,j]
                max_p = neighbors.max()
                min_p = neighbors.min()
                if (p>0):
                    zero_cross = True if min_p < 0 else False 
                else:
                    zero_cross = False if max_p > 0 else False
                if (max_p-min_p) > thres and zero_cross:
                    output[i,j] = 1 
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

        img_s = (img-offset)/scale

        def sift(imf):
            """ Apply the sifting procedure on the given 2-D signal. """
            max_envelope, min_envelope = self.envelope(imf)
            mean = (max_envelope + min_envelope) * 0.5 
            imf = imf - mean
            return imf

        n = 0 # Number of IMFs

        # Creates a tensor such each matrix represents an IMF
        IMFs = np.empty((n,) + img.shape) 
        notFinished = True


        while notFinished:
            # At the k-th iteration of the decomposition, 
            # we refer to data as the original signal after being subtracted from the k-1 generated IMFs.
            res = img_s - np.sum(IMFs[:n], axis=0)
            x = res.copy()

            k = 0 # Iterations for current IMFs
            k_h = 0 # number of consecutive iterations to compare to S number (explained below)
            flag = True
            while flag and k < self.MAX:
                # The code use the S number criterion, i.e the canidate will be elected as IMF after s consecutive runs in
                # which the difference between local extrema and zero crossing is by at most 1. 
                # S is pre-determined.
                imf  = sift(x)
                zero_crossing = EMD2D.count_zero_crossings(imf)
                local_max, local_min = self.find_local_extrema(imf)
                num_extrema = len(local_max[0]) + len(local_min[0])
                if abs(num_extrema - zero_crossing) < 1:
                    k_h = k_h + 1
                else:
                    k_h = 0

                if k_h == self.S_critertion:
                    flag = False 
                x = imf.copy()
                
            # Add the chosen canidate to the IMFs
            IMFs = np.vstack((IMFs, imf.copy()[None,:]))
            n+= 1
            if self.end_condition(img, IMFs) or (self.max_imf>0 and n>=self.max_imf):
                notFinished = False
                break
        res = img_s - np.sum(IMFs[:n], axis=0)
        if not np.allclose(res, 0):
                IMFs = np.vstack((IMFs, res[None,:]))
                n += 1
        IMFs = IMFs*scale
        IMFs[-1] += offset
        return IMFs
        



if __name__ == "__main__":
    emd = EMD2D()
    img = cv2.imread('DATA/00025.jpg',0)
    imfs = emd.EMD(img)
    print(imfs.shape)
    


    
       
                

            

