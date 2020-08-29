from EMD2D import EMD2D
import numpy as np
import pandas as pd
import cv2
import os
import platform
from scipy import interpolate, signal, ndimage


def Bilinear(img: np.ndarray, w: float, h: float):
    new_Shape = img.shape
    new_Shape[0] *= w
    new_Shape[1] *= h

    return cv2.resize(img, dsize=new_Shape, interpolation=cv2.INTER_LINEAR)


def Bicubic(img: np.ndarray, w: float, h: float):
    new_Shape = img.shape
    new_Shape[0] *= w
    new_Shape[1] *= h

    return cv2.resize(img, dsize=new_Shape, interpolation=cv2.INTER_CUBIC)


def Lanczos4(img: np.ndarray, w: float, h: float):
    new_Shape = img.shape
    new_Shape[0] *= w
    new_Shape[1] *= h

    return cv2.resize(img, dsize=new_Shape, interpolation=cv2.INTER_LANCZOS4)


def RBF(img: np.ndarray, w: float, h: float, function='gaussian'):
    empty = Rescale(img, w, h)

    def forChannel(channel: int):
        im: np.ndarray = img[:, :, channel].copy()
        x, y = im.shape
        x, y = np.meshgrid(x, y)

        return interpolate.Rbf(x, y, im, function=function)


def Rescale(img: np.ndarray, w: float, h: float) -> np.ndarray:
    new_Shape = img.shape
    new_Shape[0] *= w
    new_Shape[1] *= h

    new_img = np.zeros(new_Shape, dtype=np.uint8)
