import numpy as np
from pyhht.emd import EmpiricalModeDecomposition as EMD
import cv2
from scipy import ndimage, signal
import scipy.fft as fft
from MyKernels import Sharpen3x3, LaplacianOfGaussian5x5, Laplace3x3, LaplaceDiag3x3
from matplotlib import pyplot as plt
from matplotlib.colors import NoNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL.Image import fromarray
from datetime import datetime
import os


class EMD2D:
    IMFs: np.ndarray = np.array([])
    Rs = None
    Gs = None
    Bs = None
    NoIMFs: int = 0

    def __init__(self, image: np.ndarray, save=False):
        if image is None:
            return

        self.EMD = EMD
        self.img = None
        if save:
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

    def __call(self, imf, dtype=None) -> np.ndarray:
        if type(imf) == slice:
            start = imf.start
            if start is None:
                start = 0
            elif start < 0:
                start = 0
            elif start >= self.__len__():
                print("Slice unavailable - start problem")
                start = self.__len__() - 2

            stop = imf.stop
            if stop is None:
                stop = self.__len__()

            elif stop <= start:
                stop = start + 1

            elif stop >= self.__len__():
                stop = self.__len__()

            caller = list(range(start, stop))
            if len(self.shape) == 2:
                tmp = np.empty((len(caller), self.shape[0], self.shape[1]))
            else:
                tmp = np.empty((len(caller), self.shape[0], self.shape[1], self.shape[2]))
            ln = 0
            for i in caller:
                tmp[ln] = self.__call(i)
                ln += 1
            return tmp

        else:
            if len(self.shape) == 2:
                if imf < self.IMFs.shape[0]:
                    if dtype is None:
                        return self.IMFs[imf].transpose()
                    return self.IMFs[imf].transpose().astype(np.uint8)
                return np.zeros(self.shape).astype(np.uint8)

            if dtype == None:
                part1 = part2 = part3 = np.zeros((self.shape[0], self.shape[1]))
            else:
                part1 = part2 = part3 = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)

            x1 = dtype is None
            if imf < self.Rs.shape[0]:
                part1 = self.Rs[imf, :, :].transpose().astype(np.uint8) * (1 - x1) + \
                        x1 * self.Rs[imf, :, :].transpose()

            if imf < self.Gs.shape[0]:
                part2 = self.Gs[imf, :, :].transpose().astype(np.uint8) * (1 - x1) + \
                        x1 * self.Gs[imf, :, :].transpose()

            if imf < self.Bs.shape[0]:
                part3 = self.Bs[imf, :, :].transpose().astype(np.uint8) * (1 - x1) + x1 * self.Bs[imf, :, :].transpose()

            return cv2.merge((part1, part2, part3))

    def __call__(self, imf) -> np.ndarray:
        if type(imf) == slice:
            pass

    def __getitem__(self, imf):
        if type(imf) == slice:
            tmp = self.__call(imf=imf)
            return tmp

        elif type(imf) == int:
            tmp = self.__call(imf=imf)
            return tmp

        keys = list(imf)
        tmp = self.__call(imf=keys[0])

        len2 = len(self.shape) == 2

        keys = keys[1:]
        if len2:
            while len(keys) != 2:
                keys.append(slice(None, None))

            return tmp[:, keys[0], keys[1]]

        else:
            while len(keys) != 3:
                keys.append(slice(None, None))

            return tmp[:, keys[0], keys[1], keys[2]]

    def reConstruct(self):
        def act(Imfs: np.ndarray, axis=0):
            return np.sum(Imfs, axis=axis)

        if len(self.shape) == 2:
            return act(self.IMFs).transpose().astype(np.uint8)

        return cv2.merge((act(self.Rs).transpose().astype(np.uint8), act(self.Gs).transpose().astype(np.uint8),
                          act(self.Bs).transpose().astype(np.uint8)))

    def ForShow(self, median_filter=False, sharp=False):
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

    def __copy(self):
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

    def __cmp__(self, other):
        if other.shape != self.shape:
            print("Couldn't compare")
            return False

        x1 = self.ForShow(False)
        if type(other) == np.ndarray:
            return x1 == other

        x2 = other.ForShow(False)
        return x1 == x2

    def applyFilter(self, **kwargs):
        pass

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

    def applyGaussian(self, sigma=0.1, median_filter=False):
        dx = self.ForShow(median_filter=median_filter)
        dx = ndimage.gaussian_filter(dx, sigma)
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
            # filtered.imshow(self.ForShow(), cmap='gray', norm=NoNorm())
            # filtered.set_title('Reconstructed & Median-Filtered picture')

        else:
            fig, (origin, decomp, filtered) = plt.subplots(3, 1, figsize=(20, 20))
            fig.suptitle('All picture forms')
            origin.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            origin.set_title('Original')
            decomp.imshow(self.ForShow(False))
            decomp.set_title('Reconstructed picture')
            # filtered.imshow(self.ForShow())
            # filtered.set_title('Reconstructed & Median-Filtered picture')

        plt.show()

    def surfaces(self):
        x0, y0 = self.shape
        x0, y0 = np.meshgrid(range(y0), range(x0))

        if len(self.shape) == 2:
            n = self.NoIMFs
            fig, plots = plt.subplots(nrows=n, ncols=1, figsize=(20, 20))

            for i in range(n):
                plots[i] = plt.axes(projection='3d')
                if i == n - 1:
                    plots[i].set_title('Residue')
                else:
                    plots[i].set_title('IMF ' + str(i + 1))
                plots[i].plot_surface(x0, y0, self.__getitem__(i), cmap='binary', norm=NoNorm())
                plots[i].grid()

            plt.show()

        else:
            n = max(self.Rs.shape[0], self.Bs.shape[0], self.Gs.shape[0])
            fig, plots = plt.subplots(nrows=n, ncols=3, figsize=(20, 20))

            for i in range(n):
                plots[i, 0] = plots[i, 1] = plots[i, 2] = plt.axes(projection='3d')
                if i == n - 1:
                    plots[i][0].set_title('Red Residue')
                    plots[i][1].set_title('Green Residue')
                    plots[i][2].set_title('Blue Residue')

                else:
                    plots[i][0].set_title('Red IMF ' + str(i + 1))
                    plots[i][1].set_title('Green IMF ' + str(i + 1))
                    plots[i][2].set_title('Blue IMF ' + str(i + 1))

                surf = self.__getitem__(i)

                plots[i][0].plot_surface(x0, y0, surf[:, :, 0], cmap='binary', norm=NoNorm())
                plots[i][0].grid()

                plots[i][1].plot_surface(x0, y0, surf[:, :, 1], cmap='binary', norm=NoNorm())
                plots[i][1].grid()

                plots[i][2].plot_surface(x0, y0, surf[:, :, 2], cmap='binary', norm=NoNorm())
                plots[i][2].grid()
            plt.show()

    def copy(self):
        return self.__copy()

    def save(self, with_imfs=True):
        tmp = self.reConstruct()
        now = datetime.now()
        curdir = os.getcwd()
        curdir = curdir.replace(curdir[2], '/') + '/Edited Data/' + now.strftime("%d-%m-%Y%H-%M-%S")
        os.mkdir(curdir)
        curdir = 'Edited Data/' + now.strftime("%d-%m-%Y%H-%M-%S") + '/'
        if with_imfs:
            for i in range(self.__len__()):
                tmp1 = self.__getitem__(i)
                cv2.imwrite(curdir + 'IMF_' + str(i+1) + '.jpg', tmp1)

        try:
            cv2.imwrite(curdir + 'Original.jpg', tmp)
        except Exception:
            print("Can't save")
            return False
        return True