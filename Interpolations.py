import numpy as np
import cv2
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
    empty = RescaleAndScatter(img, w, h)

    def forChannel(channel: int):
        im: np.ndarray = img[:, :, channel].copy()
        x, y = im.shape
        x, y = np.meshgrid(x, y)

        return interpolate.Rbf(x, y, im, function=function)


def RescaleAndScatter(img: np.ndarray, w: float, h: float) -> np.ndarray:
    epsilon = 0.1
    new_Shape = [0, 0]
    new_Shape[0] = int(w * img.shape[0])
    new_Shape[1] = int(h * img.shape[1])

    # new_img: np.ndarray = np.ones(tuple(new_Shape)) * -1
    new_img: np.ndarray = np.ones(tuple(new_Shape), dtype=np.uint8) * 0

    to_interp_x: np.ndarray = np.arange(0, img.shape[0]) * w
    to_interp_y: np.ndarray = np.arange(0, img.shape[1]) * h

    spots_x = np.abs(to_interp_x - np.round(to_interp_x)) <= epsilon
    spots_y = np.abs(to_interp_y - np.round(to_interp_y)) <= epsilon

    spots_x += ~spots_x
    spots_y += ~spots_y

    for_original_x = np.round(to_interp_x[spots_x].copy() / w).astype(int)
    for_original_y = np.round(to_interp_y[spots_y].copy() / h).astype(int)

    for_original_x, for_original_y = np.meshgrid(for_original_x, for_original_y)

    to_interp_x = np.round(to_interp_x[spots_x]).astype(int)
    to_interp_y = np.round(to_interp_y[spots_y]).astype(int)

    to_interp_x, to_interp_y = np.meshgrid(to_interp_x, to_interp_y)

    new_img[to_interp_x, to_interp_y] = img[for_original_x, for_original_y]

    # new_img[new_img > 0] = 1
    # print(new_img.sum())
    # new_img[new_img > 0] = 255
    x = Image.fromarray(new_img)
    x.show()
    # img[img > 0] = 1
    # print(img.sum())


image = cv2.imread('DATA/dog.jpg', 0)
RescaleAndScatter(image, 3.55, 2.2)
