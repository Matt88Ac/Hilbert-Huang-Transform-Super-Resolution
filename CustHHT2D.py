from PIL.Image import fromarray
import cv2
import numpy as np
from scipy import ndimage, interpolate
from skimage.morphology import local_maxima, local_minima


class EMD2D:
    imfs: np.ndarray = np.array([])
    NoIMFS: int = 0

    def __init__(self, img: np.ndarray):
        img_min, img_max = img.min(), img.max()
        scale = img_max - img_min
        scaled_img = (img.copy() - img_min) / scale

    @classmethod
    def envelope(cls, img: np.ndarray):
        loc_min = local_minima(img)
        loc_max = local_maxima(img)
        max_points = local_maxima(img, indices=True)
        min_points = local_minima(img, indices=True)

        px_max = max_points[0]
        py_max = max_points[1]

        px_min = min_points[0]
        py_min = min_points[1]

        def spline(X, Y, Z):
            return interpolate.interp2d(X, Y, Z, kind='cubic')

        splineMax = spline(px_max, py_max, img[loc_max])
        splineMin = spline(px_min, py_min, img[loc_min])

        newx = np.arange(0, img.shape[0], 0.01)
        newy = np.arange(0, img.shape[1], 0.01)

        newx, newy = np.meshgrid(newx, newy)

        return splineMax(newx, newy), splineMin(newx, newy)
