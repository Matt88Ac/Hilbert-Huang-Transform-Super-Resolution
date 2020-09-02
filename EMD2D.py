import numpy as np
from pyhht.emd import EmpiricalModeDecomposition as EMD
import cv2
from scipy import ndimage, signal
import scipy.fft as fft
from MyKernels import Sharpen3x3, LaplacianOfGaussian5x5, Laplace3x3, LaplaceDiag3x3
from matplotlib import pyplot as plt
from matplotlib.colors import NoNorm
from PIL.Image import fromarray


class EMD2D:
    IMFs: np.ndarray = np.array([])
    Rs = None
    Gs = None
    Bs = None
    NoIMFs: int = 0

    def __init__(self, image: np.ndarray):
        if image is None:
            return

        self.EMD = EMD
        self.img = image

        self.shape = image.shape

        def emd_images_col(colOfImage: np.ndarray):
            return self.EMD(colOfImage).decompose()

        def concatZeros(row: np.ndarray, howMuch: int, bo) -> np.ndarray:
            if len(row.shape) == 1:
                return np.vstack((row, np.zeros((howMuch, row.shape[0]))))
            return np.vstack((row, np.zeros((howMuch, row.shape[1]))))

        def checkZeroPad(imfs: np.ndarray):
            return self.NoIMFs - imfs.shape[0]

        def AddDecomposed(newIMF: np.ndarray):
            newArr = np.array([np.vstack((self.IMFs[0], newIMF[0]))])
            n = checkZeroPad(newIMF)
            for i in range(1, min(self.NoIMFs, newIMF.shape[0])):
                newArr = np.vstack((newArr, np.array([np.vstack((self.IMFs[i], newIMF[i]))])))
            if n == 0:
                self.IMFs = newArr.copy()
                return
            elif n < 0:
                for i in range(self.NoIMFs, newIMF.shape[0]):
                    toAdd = newArr[0].shape[0] - 1
                    ta = np.array([concatZeros(newIMF[i], toAdd, False)])
                    newArr = np.vstack((newArr, ta))
                self.IMFs = newArr.copy()
                self.NoIMFs = newIMF.shape[0]
                return

            for i in range(newIMF.shape[0], self.NoIMFs):
                ta = np.array([concatZeros(self.IMFs[i], 1, True)])
                newArr = np.vstack((newArr, ta))
            self.IMFs = newArr.copy()

        def Run(img: np.ndarray) -> np.ndarray:
            deco = emd_images_col(img[:, 0])
            self.NoIMFs = deco.shape[0]
            self.IMFs = deco.copy()

            for i in range(1, img.shape[1]):
                deco = emd_images_col(img[:, i])
                AddDecomposed(deco.copy())
            return self.IMFs.copy()

        if len(image.shape) == 3:
            No = 0
            self.Rs = Run(image[:, :, 0])
            No += self.NoIMFs
            self.Gs = Run(image[:, :, 1])
            No += self.NoIMFs
            self.Bs = Run(image[:, :, 2])
            self.NoIMFs += No
        else:
            Run(image)

    def __getitem__(self, imf):
        if len(self.shape) == 2:
            if imf < self.IMFs.shape[0]:
                return self.IMFs[imf].transpose().astype(np.uint8)
            return np.zeros(self.shape).astype(np.uint8)

        part1 = part2 = part3 = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)

        if imf < self.Rs.shape[0]:
            part1 = self.Rs[imf, :, :].transpose().astype(np.uint8)

        if imf < self.Gs.shape[0]:
            part2 = self.Gs[imf, :, :].transpose().astype(np.uint8)

        if imf < self.Bs.shape[0]:
            part3 = self.Bs[imf, :, :].transpose().astype(np.uint8)

        return cv2.merge((part1, part2, part3))

    def reConstruct(self):
        def act(Imfs: np.ndarray, axis=0):
            return np.sum(Imfs, axis=axis)

        if len(self.shape) == 2:
            return act(self.IMFs).transpose().astype(np.uint8)

        return cv2.merge((act(self.Rs).transpose().astype(np.uint8), act(self.Gs).transpose().astype(np.uint8),
                          act(self.Bs).transpose().astype(np.uint8)))

    def ForShow(self, median_filter=True, sharp=False):
        if len(self.shape) == 2:
            ret = self.reConstruct()
            if median_filter:
                ret = ndimage.median_filter(ret, 3)

            if sharp:
                ret = signal.convolve2d(ret, Sharpen3x3, mode='same')

            return ret

        ret = cv2.cvtColor(self.reConstruct(), cv2.COLOR_BGR2RGB)
        if median_filter:
            ret = ndimage.median_filter(ret, 3)

        if sharp:
            ret[:, :, 0] = signal.convolve2d(ret[:, :, 0], Sharpen3x3, mode='same')
            ret[:, :, 1] = signal.convolve2d(ret[:, :, 1], Sharpen3x3, mode='same')
            ret[:, :, 2] = signal.convolve2d(ret[:, :, 2], Sharpen3x3, mode='same')

        return ret

    def __len__(self):
        if len(self.shape) == 2:
            return self.IMFs.shape[0]
        return max(self.Rs.shape[0], self.Bs.shape[0], self.Gs.shape[0])

    def __copy__(self):
        tmp = EMD2D(image=None)
        tmp.shape = self.shape
        if not self.Gs:
            tmp.IMFs = self.IMFs.copy()

        if self.Gs:
            tmp.Gs = self.Gs.copy()
            tmp.Bs = self.Bs.copy()
            tmp.Rs = self.Rs.copy()

        tmp.NoIMFs = self.NoIMFs
        return tmp

    def __repr__(self):
        tmp = self.ForShow(median_filter=False)
        if len(self.shape) == 2:
            plt.imshow(tmp, cmap='gray', norm=NoNorm())
        else:
            plt.imshow(tmp)
        plt.show()
        return ""

    def __hash__(self):
        tmp = self.reConstruct()
        if len(self.shape) == 2:
            plt.imshow(tmp, cmap='gray', norm=NoNorm())
        else:
            plt.imshow(tmp)
        plt.show()
        return 0

    def applyLoG5x5(self, median_filter=False):
        dx = self.ForShow(median_filter=median_filter)
        if len(self.shape) == 2:
            return signal.convolve2d(dx, LaplacianOfGaussian5x5, mode='same')
        for i in range(3):
            dx[:, :, i] = signal.convolve2d(dx[:, :, i], LaplacianOfGaussian5x5, mode='same')
        return dx

    def applyLaplace3x3(self, median_filter=False):
        dx = self.ForShow(median_filter=median_filter)
        if len(self.shape) == 2:
            return signal.convolve2d(dx, Laplace3x3, mode='same')
        for i in range(3):
            dx[:, :, i] = signal.convolve2d(dx[:, :, i], Laplace3x3, mode='same')
        return dx

    def applyDiagLaplace3x3(self, median_filter=False):
        dx = self.ForShow(median_filter=median_filter)
        if len(self.shape) == 2:
            return signal.convolve2d(dx, LaplaceDiag3x3, mode='same')
        for i in range(3):
            dx[:, :, i] = signal.convolve2d(dx[:, :, i], LaplaceDiag3x3, mode='same')
        return dx

    def applyFFT(self, median_filter=False, as_int=False):
        dx = self.ForShow(median_filter=median_filter)
        f1 = fft.fft(dx)

        return f1.real * (1 - int(as_int)) + f1.real.astype(np.uint8) * int(as_int), f1

    def compare(self):
        if len(self.shape) == 2:
            fig, (origin, decomp, filtered) = plt.subplots(3, 1, figsize=(20, 20))
            fig.suptitle('All picture forms')
            origin.imshow(self.img, cmap='gray', norm=NoNorm())
            origin.set_title('Original')
            decomp.imshow(self.ForShow(False), cmap='gray', norm=NoNorm())
            decomp.set_title('Reconstructed picture')
            filtered.imshow(self.ForShow(), cmap='gray', norm=NoNorm())
            filtered.set_title('Reconstructed & Median-Filtered picture')

        else:
            fig, (origin, decomp, filtered) = plt.subplots(3, 1, figsize=(20, 20))
            fig.suptitle('All picture forms')
            origin.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            origin.set_title('Original')
            decomp.imshow(self.ForShow(False))
            decomp.set_title('Reconstructed picture')
            filtered.imshow(self.ForShow())
            filtered.set_title('Reconstructed & Median-Filtered picture')

        plt.show()
