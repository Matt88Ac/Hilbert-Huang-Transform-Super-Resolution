import numpy as np
from pyhht.emd import EmpiricalModeDecomposition as EMD
import cv2
from scipy import ndimage, signal
from matplotlib import pyplot as plt
from matplotlib.colors import NoNorm
from PIL.Image import fromarray, Image
from datetime import datetime
import os
from General_Scripts import Sharpen3x3
from General_Scripts import interactiveImread, imread


class EMD2D:
    IMFs: np.ndarray = np.array([])
    Rs = None
    Gs = None
    Bs = None
    NoIMFs: int = 0

    def __init__(self, image):
        if image is None:
            return
        self.EMD = EMD
        if type(image) == Image:
            self.img = np.array(image)

        else:
            self.img = image.copy()
        self.shape = self.img.shape

        self.iter = 0

        def emd_images_col(colOfImage: np.ndarray) -> np.ndarray:
            return self.EMD(colOfImage).decompose()

        def concatZeros(row: np.ndarray, howMuch: int) -> np.ndarray:
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
                    ta = np.array([concatZeros(newIMF[i], toAdd)])
                    newArr = np.vstack((newArr, ta))
                self.IMFs = newArr.copy()
                self.NoIMFs = newIMF.shape[0]
                return

            for i in range(newIMF.shape[0], self.NoIMFs):
                ta = np.array([concatZeros(self.IMFs[i], 1)])
                newArr = np.vstack((newArr, ta))
            self.IMFs = newArr.copy()

        def newAdder(newIMF: np.ndarray):
            n = checkZeroPad(newIMF)
            if n == 0:
                self.IMFs = np.concatenate((self.IMFs, newIMF), axis=1)

            elif n < 0:
                tempo = np.zeros((abs(n), self.IMFs.shape[1], self.IMFs.shape[2]))
                self.IMFs = np.concatenate((self.IMFs, tempo), axis=0)
                self.IMFs = np.concatenate((self.IMFs, newIMF), axis=1)
                self.NoIMFs = self.IMFs.shape[0]

            else:
                tempo = np.zeros((abs(n), newIMF.shape[1], newIMF.shape[2]))
                tempo = np.concatenate((newIMF, tempo))
                self.IMFs = np.concatenate((self.IMFs, tempo), axis=1)

        def Run(img: np.ndarray) -> np.ndarray:
            deco = emd_images_col(img[:, 0])
            self.NoIMFs = deco.shape[0]
            self.IMFs = deco.copy().reshape((deco.shape[0], 1, deco.shape[1]))

            for i in range(1, img.shape[1]):
                deco = emd_images_col(img[:, i])
                deco = deco.reshape((deco.shape[0], 1, deco.shape[1]))
                newAdder(deco.copy())
            return self.IMFs.copy()

        if len(self.img.shape) == 3:
            No = 3
            self.Rs = Run(self.img[:, :, 0])
            No += self.NoIMFs
            errorImf = self.img[:, :, 0] - np.sum(self.Rs, axis=0).transpose().astype(np.uint8)
            self.Rs = np.concatenate((self.Rs, errorImf.transpose()[None]))

            self.Gs = Run(self.img[:, :, 1])
            No += self.NoIMFs
            errorImf = self.img[:, :, 1] - np.sum(self.Gs, axis=0).transpose().astype(np.uint8)
            self.Gs = np.concatenate((self.Gs, errorImf.transpose()[None]))

            self.Bs = Run(self.img[:, :, 2])
            self.NoIMFs += No
            errorImf = self.img[:, :, 2] - np.sum(self.Bs, axis=0).transpose().astype(np.uint8)
            self.Bs = np.concatenate((self.Bs, errorImf.transpose()[None]))

        else:
            Run(self.img)
            errorImf = self.img - self.reConstruct()
            errorImf = errorImf.transpose()
            self.IMFs = np.concatenate((self.IMFs, errorImf[None]))
            self.NoIMFs += 1

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

            if dtype is None:
                part1 = part2 = part3 = np.zeros((self.shape[0], self.shape[1]))
            else:
                part1 = part2 = part3 = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)

            x1 = dtype == None
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
            tmp = self.__call(imf=imf, dtype=0)
            return tmp

        elif type(imf) == int:
            tmp = self.__call(imf=imf, dtype=0)
            return tmp

        keys = list(imf)
        tmp = self.__call(imf=keys[0], dtype=0)

        len2 = len(self.shape) == 2

        keys = keys[1:]
        if len2:
            while len(keys) != 2:
                keys.append(slice(None, None))

            if len(tmp.shape) == 3:
                return tmp[:, keys[0], keys[1]]
            return tmp[keys[0], keys[1]]

        else:
            while len(keys) != 3:
                keys.append(slice(None, None))
            if len(tmp.shape) == 4:
                return tmp[:, keys[0], keys[1], keys[2]]
            return tmp[keys[0], keys[1], keys[2]]

    def __getitem__(self, imf) -> np.ndarray:
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

            if len(tmp.shape) == 3:
                return tmp[:, keys[0], keys[1]]
            return tmp[keys[0], keys[1]]

        else:
            while len(keys) != 3:
                keys.append(slice(None, None))
            if len(tmp.shape) == 4:
                return tmp[:, keys[0], keys[1], keys[2]]
            return tmp[keys[0], keys[1], keys[2]]

    def __assemble(self, dtype=None) -> np.ndarray:
        if dtype is None:
            if len(self.shape) == 2:
                return np.sum(self.IMFs, axis=0).transpose()

            R = np.sum(self.Rs, axis=0).transpose()
            G = np.sum(self.Gs, axis=0).transpose()
            B = np.sum(self.Bs, axis=0).transpose()

            return cv2.merge((R, G, B))

        else:
            if len(self.shape) == 2:
                return np.sum(self.IMFs, axis=0).transpose().astype(dtype)

            R = np.sum(self.Rs, axis=0).transpose().astype(dtype)
            G = np.sum(self.Gs, axis=0).transpose().astype(dtype)
            B = np.sum(self.Bs, axis=0).transpose().astype(dtype)

            return cv2.merge((R, G, B))

    def reConstruct(self) -> np.ndarray:
        return self.__assemble()

    def ForShow(self, median_filter=False):
        if len(self.shape) == 2:
            ret = self.__assemble(dtype=np.uint8)
            if median_filter:
                ret = ndimage.median_filter(ret, 3)
            return ret

        ret = cv2.cvtColor(self.__assemble(dtype=np.uint8), cv2.COLOR_BGR2RGB)
        if median_filter:
            ret = ndimage.median_filter(ret, 3)

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
        # tmp = cv2.resize(tmp, (tmp.shape[1] * 5, tmp.shape[0] * 5), interpolation=cv2.INTER_CUBIC)
        if len(self.shape) == 2:
            plt.imshow(tmp, cmap='gray', norm=NoNorm())
        else:
            plt.imshow(tmp)
        plt.show()
        return ""

    def __cmp__(self, other):
        other1 = other
        if type(other) == Image:
            other1 = np.array(other)

        if other1.shape != self.shape:
            print("Couldn't compare")
            return False
        x1 = self.reConstruct()
        if type(other1) == np.ndarray:
            return x1 == other1

        x2 = other1.ForShow(False)
        return x1 == x2

    def __iter__(self):
        if self.iter >= len(self):
            self.iter = 0

        while self.iter < len(self):
            self.iter += 1
            yield self[self.iter - 1]

        self.iter = 0

    def applyFilter(self, **kwargs):
        temp = self.ForShow()
        keys = list(kwargs.keys())

        gaussian = 'gaussian' in keys
        sigma = 'sigma' in keys
        LoG = 'LoG' in keys
        sobel = 'sobel' in keys
        prewitt = 'prewitt' in keys
        laplace = 'laplace' in keys
        sharp = 'sharpen' in keys
        median = 'median' in keys
        mx = 'max' in keys
        mn = 'min' in keys
        uni = 'uniform' in keys
        spline = 'spline' in keys
        order = 'order' in keys

        if not sigma:
            sigma = 0.1
        else:
            sigma = kwargs['sigma']

        if not order:
            order = 3
        else:
            order = kwargs['order']

        if gaussian:
            temp = ndimage.gaussian_filter(temp, sigma)

        if laplace:
            temp = ndimage.laplace(temp)

        if LoG:
            temp = ndimage.gaussian_laplace(temp, sigma)

        if median:
            temp = ndimage.median_filter(temp, 3)

        if mx:
            temp = ndimage.maximum_filter(temp, 3)

        if mn:
            temp = ndimage.minimum_filter(temp, 3)

        if uni:
            temp = ndimage.uniform_filter(temp)

        if sobel:
            temp = ndimage.sobel(temp)

        if prewitt:
            temp = ndimage.prewitt(temp)

        if spline:
            temp = ndimage.spline_filter(temp, order=order)

        if sharp:
            if len(temp.shape) == 2:
                return signal.convolve2d(temp, Sharpen3x3, mode='same')
            temp[:, :, 0] = signal.convolve2d(temp[:, :, 0], Sharpen3x3, mode='same')
            temp[:, :, 1] = signal.convolve2d(temp[:, :, 1], Sharpen3x3, mode='same')
            temp[:, :, 2] = signal.convolve2d(temp[:, :, 2], Sharpen3x3, mode='same')

        return temp

    def applyFFT(self, as_int=False):
        dx = self.ForShow()
        f1 = np.fft.fft2(dx)
        if as_int:
            return f1.real.astype(np.uint8), f1

        return f1.real, f1

    def compare(self):
        if len(self.shape) == 2:
            fig, (origin, decomp) = plt.subplots(2, 1, figsize=(20, 20))
            fig.suptitle('All picture forms')
            origin.imshow(self.img, cmap='gray', norm=NoNorm())
            origin.set_title('Original')
            decomp.imshow(self.ForShow(False), cmap='gray', norm=NoNorm())
            decomp.set_title('Reconstructed picture')
            # filtered.imshow(self.ForShow(), cmap='gray', norm=NoNorm())
            # filtered.set_title('Reconstructed & Median-Filtered picture')

        else:
            fig, (origin, decomp) = plt.subplots(2, 1, figsize=(20, 20))
            fig.suptitle('All picture forms')
            origin.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            origin.set_title('Original')
            decomp.imshow(self.ForShow(False))
            decomp.set_title('Reconstructed picture')
            # filtered.imshow(self.ForShow())
            # filtered.set_title('Reconstructed & Median-Filtered picture')

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
                cv2.imwrite(curdir + 'IMF_' + str(i + 1) + '.jpg', tmp1)

        try:
            cv2.imwrite(curdir + 'Original.jpg', tmp)
        except Exception:
            print("Can't save")
            return False
        return True

    def show(self):
        x0 = fromarray(self.ForShow())
        x0.show()
