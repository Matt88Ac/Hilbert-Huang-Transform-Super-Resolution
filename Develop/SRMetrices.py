import numpy as np
from skimage.metrics import peak_signal_noise_ratio, variation_of_information, structural_similarity, \
    normalized_root_mse, adapted_rand_error


def PSNR(expected: np.ndarray, estimated: np.ndarray):
    if expected.shape == estimated.shape:
        return peak_signal_noise_ratio(expected, estimated)


def Valid_Of_Info(expected: np.ndarray, estimated: np.ndarray):
    if expected.shape == estimated.shape:
        return variation_of_information(expected, estimated)


def Normalized_RMSE(expected: np.ndarray, estimated: np.ndarray):
    if expected.shape == estimated.shape:
        return normalized_root_mse(expected, estimated)


def SSIM(expected: np.ndarray, estimated: np.ndarray):
    if expected.shape == estimated.shape:
        return structural_similarity(expected, estimated)


def Adapted_Rand_Error(expected: np.ndarray, estimated: np.ndarray):
    if expected.shape == estimated.shape:
        return adapted_rand_error(expected, estimated)
