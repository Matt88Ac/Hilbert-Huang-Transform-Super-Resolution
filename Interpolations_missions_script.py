from EMD2D import EMD2D
import numpy as np
import pandas as pd
import cv2
import os
import platform
from scipy import interpolate, signal, ndimage
from tensorflow import image
from matplotlib.pyplot import imshow
from PIL import Image


def Gaussian(img: np.ndarray, w: float, h: float):
    new_Shape = [0, 0]
    new_Shape[0] = int(w * img.shape[0])
    new_Shape[1] = int(h * img.shape[1])

    return image.resize(img, size=new_Shape, method=image.ResizeMethod.GAUSSIAN).numpy().astype(np.uint8)


def MitchelCubic(img: np.ndarray, w: float, h: float):
    new_Shape = [0, 0]
    new_Shape[0] = int(w * img.shape[0])
    new_Shape[1] = int(h * img.shape[1])

    return image.resize(img, size=new_Shape, method=image.ResizeMethod.MITCHELLCUBIC).numpy().astype(np.uint8)


# img = cv2.imread('DATA/dog.jpg')
# x1 = Image.fromarray(Gaussian(img, 2, 2))
# x1.show()


def Bilinear(img: np.ndarray, w: float, h: float):
    new_Shape = [0, 0]
    new_Shape[0] = int(w * img.shape[0])
    new_Shape[1] = int(h * img.shape[1])

    return cv2.resize(img, dsize=tuple(new_Shape), interpolation=cv2.INTER_LINEAR)


def Bicubic(img: np.ndarray, w: float, h: float):
    new_Shape = [0, 0]
    new_Shape[0] = int(w * img.shape[0])
    new_Shape[1] = int(h * img.shape[1])

    return cv2.resize(img, dsize=tuple(new_Shape), interpolation=cv2.INTER_CUBIC)


def Lanczos4(img: np.ndarray, w: float, h: float):
    new_Shape = [0, 0]
    new_Shape[0] = int(w * img.shape[0])
    new_Shape[1] = int(h * img.shape[1])

    return cv2.resize(img, dsize=tuple(new_Shape), interpolation=cv2.INTER_LANCZOS4)


def RBF(img: np.ndarray, w: float, h: float, function='gaussian'):
    empty = Rescale(img, w, h)

    def forChannel(channel: int):
        im: np.ndarray = img[:, :, channel].copy()
        x, y = im.shape
        x, y = np.meshgrid(x, y)

        return interpolate.Rbf(x, y, im, function=function)


def Rescale(img: np.ndarray, w: float, h: float) -> np.ndarray:
    new_Shape = [0, 0]
    new_Shape[0] = int(w * img.shape[0])
    new_Shape[1] = int(h * img.shape[1])

    new_img = np.zeros(tuple(new_Shape), dtype=np.uint8)
