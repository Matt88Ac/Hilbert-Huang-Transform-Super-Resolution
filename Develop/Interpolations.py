import numpy as np
import cv2
from scipy import interpolate, signal, ndimage
from tensorflow import image
from matplotlib.pyplot import imshow
from PIL import Image
from Develop.EMD2D import EMD2D


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


def Lanczos3(img: np.ndarray, w: float, h: float):
    new_Shape = [0, 0]
    new_Shape[0] = int(w * img.shape[0])
    new_Shape[1] = int(h * img.shape[1])

    return image.resize(img, size=new_Shape, method=image.ResizeMethod.LANCZOS3).numpy().astype(np.uint8)


def Lanczos5(img: np.ndarray, w: float, h: float):
    new_Shape = [0, 0]
    new_Shape[0] = int(w * img.shape[0])
    new_Shape[1] = int(h * img.shape[1])

    return image.resize(img, size=new_Shape, method=image.ResizeMethod.LANCZOS5).numpy().astype(np.uint8)


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


def_interpolations = [Gaussian, Bicubic, Bilinear, Lanczos5, Lanczos3, Lanczos4, MitchelCubic]


# ********************************************************************************************************** #


def RBF(img: np.ndarray, w: float, h: float, function='gaussian'):
    empty: np.ndarray = RescaleAndScatter(img, w, h)
    xx, yy = np.meshgrid(range(img.shape[0]), range(img.shape[1]))
    interp = interpolate.Rbf(xx, yy, img, function=function, smooth=0.001)

    # XI = np.linspace(0, img.shape[0], int(img.shape[0] * w))
    # YI = np.linspace(0, img.shape[1], int(img.shape[1] * h))

    # XI, YI = np.meshgrid(XI, YI)
    # res = interp(XI, YI)

    # res = Image.fromarray(res)
    # res.show()


def RescaleAndScatter(img: np.ndarray, w: float, h: float) -> np.ndarray:
    epsilon = 0.1
    new_Shape = [0, 0]
    new_Shape[0] = int(w * img.shape[0])
    new_Shape[1] = int(h * img.shape[1])

    new_img: np.ndarray = np.ones(tuple(new_Shape)) * -1
    # new_img: np.ndarray = np.ones(tuple(new_Shape), dtype=np.uint8) * 0

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

    return new_img

    # new_img[new_img > 0] = 1
    # print(new_img.sum())
    # new_img[new_img > 0] = 255
    # x = Image.fromarray(new_img)
    # x.show()
    # img[img > 0] = 1
    # print(img.sum())


def imreadAndEMD(name: str, grey=0):
    img = cv2.imread('DATA/' + name, grey)

    return EMD2D(image=img), img

# image = cv2.imread('DATA/dog.jpg', 0)
# image = cv2.resize(image, (70, 70), interpolation=cv2.INTER_CUBIC)
# image = Image.fromarray(image)
# image.show()
# RescaleAndScatter(image, 3.55, 2.2)
# RBF(image, 5, 2)
