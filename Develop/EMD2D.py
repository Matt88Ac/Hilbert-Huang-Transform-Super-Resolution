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

    def __init__(self, image: np.ndarray):
        self.IMFs: np.ndarray = np.array([])
        self.Rs = None
        self.Gs = None
        self.Bs = None
        self.NoIMFs: int = 0

        if image is None:
            return
        self.EMD = EMD
        self.img = image

        self.shape = image.shape
        self.MeanFrequency = np.array([])
        self.stdFrequency = self.MeanFrequency.copy()

        def Run(img: np.ndarray):
            self.IMFs: np.ndarray = np.array([])
            self.MeanFrequency = np.array([])
            self.stdFrequency = self.MeanFrequency.copy()

            def emd_images_col(colOfImage: np.ndarray):
                to_ret = self.EMD(colOfImage).decompose()
                return to_ret  # .reshape((to_ret.shape[1], to_ret.shape[0]))

            def fftMyIMF(imf: np.ndarray) -> np.ndarray:
                return np.fft.fft(imf).real

            def decAndFFT(colOfImage: np.ndarray) -> np.ndarray:
                decCol = emd_images_col(colOfImage)
                dft = fftMyIMF(decCol)
                self.MeanFrequency = np.append(self.MeanFrequency, dft.mean(axis=1))
                self.stdFrequency = np.append(self.stdFrequency, dft.std(axis=1))
                return decCol

            for i in range(img.shape[1]):
                newImf = decAndFFT(img[:, i])
                if len(self.IMFs) == 0:
                    self.IMFs = newImf.copy().reshape((newImf.shape[0], 1, newImf.shape[1]))
                    self.NoIMFs = self.IMFs.shape[0]
                    continue

                diff = self.NoIMFs - newImf.shape[0]
                newImf = newImf.reshape((newImf.shape[0], 1, newImf.shape[1]))
                if diff == 0:
                    self.IMFs = np.concatenate((self.IMFs, newImf), axis=1)
                    continue

                elif diff < 0:
                    tempo = np.zeros((abs(diff), self.IMFs.shape[1], self.IMFs.shape[2]))
                    self.IMFs = np.concatenate((self.IMFs, tempo), axis=0)
                    self.IMFs = np.concatenate((self.IMFs, newImf), axis=1)
                    self.NoIMFs = self.IMFs.shape[0]

                else:
                    tempo = np.zeros((abs(diff), newImf.shape[1], newImf.shape[2]))
                    tempo = np.concatenate((newImf, tempo))
                    self.IMFs = np.concatenate((self.IMFs, tempo), axis=1)

            mx = self.MeanFrequency.max()
            mn = self.MeanFrequency.min()

            diffs = (mx - mn) / self.NoIMFs
            indicator = np.empty((self.NoIMFs, 2))
            indicator[0] = np.array([mn, mn + diffs])
            newImf = None

            for i in range(1, self.NoIMFs):
                indicator[i] = np.array([indicator[i - 1][1], diffs + indicator[i - 1][1]])

            # indicator = indicator.reshape(1, indicator.shape[0], indicator.shape[1])
            mn = mx = None
            diffs = None
            diff = None

            newImf = np.zeros(self.IMFs.shape)
            for i in range(self.img.shape[1]):
                tempo: np.ndarray = self.IMFs[:, i, :].copy()
                tempo = tempo[np.where(tempo.any(axis=1))[0]].transpose()
                imfsFreqs = np.fft.fft(tempo).real.mean(axis=0)
                if tempo.shape[1] == self.img.shape[1]:
                    continue
                pass

                #   finder = np.repeat(indicator, tempo.shape[1], axis=0)
                #  tempo = np.where(finder[:, :, 0] <= imfsFreqs < finder[:, :, 1], imfsFreqs)
                finder = np.zeros(self.NoIMFs)
                for j in range(self.NoIMFs):
                    finder[j] += ((indicator[j, 0] <= imfsFreqs) & (imfsFreqs <= indicator[j, 1])).sum()
                    if j == 0:
                        finder[j] += (imfsFreqs < indicator[j, 0]).sum()
                    elif j == self.NoIMFs - 1:
                        finder[j] += (imfsFreqs > indicator[j, 1]).sum()
                    kkk = (j - 1) % self.NoIMFs
                    while finder[j] > 1:
                        if finder[kkk] >= 1:
                            kkk = (kkk - 1) % self.NoIMFs
                            continue
                        finder[kkk] = 1
                        finder[j] -= 1
                        kkk = (kkk - 1) % self.NoIMFs

                finder = finder > 0
                newImf[finder, i, :] = tempo.transpose().copy()

            self.IMFs = newImf.copy()
            return self.IMFs, self.MeanFrequency, self.stdFrequency

        if len(self.shape) == 2:
            Run(image)
            errorImf = (self.img - self.reConstruct()).transpose()
            errorImf = errorImf.reshape((1, errorImf.shape[0], errorImf.shape[1]))
            self.IMFs = np.concatenate((self.IMFs, errorImf), axis=0)
            self.NoIMFs += 1

        else:
            No = 3
            self.Rs, m1, s1 = Run(self.img[:, :, 0])
            No += self.NoIMFs
            errorImf = self.img[:, :, 0] - np.sum(self.Rs, axis=0).transpose().astype(np.uint8)
            self.Rs = np.concatenate((self.Rs, errorImf.transpose()[None]))

            self.Gs, m2, s2 = Run(self.img[:, :, 1])
            No += self.NoIMFs
            errorImf = self.img[:, :, 1] - np.sum(self.Gs, axis=0).transpose().astype(np.uint8)
            self.Gs = np.concatenate((self.Gs, errorImf.transpose()[None]))

            self.Bs, m3, s3 = Run(self.img[:, :, 2])
            self.NoIMFs += No
            errorImf = self.img[:, :, 2] - np.sum(self.Bs, axis=0).transpose().astype(np.uint8)
            self.Bs = np.concatenate((self.Bs, errorImf.transpose()[None]))

            self.stdFrequency = (s1, s2, s3)
            self.MeanFrequency = (m1, m2, m3)

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
                tmp[ln] = self.__call(i, dtype=dtype)
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
                if x1:
                    part1 = self.Rs[imf, :, :].transpose()
                else:
                    part1 = self.Rs[imf, :, :].transpose().astype(np.uint8)

            if imf < self.Gs.shape[0]:
                if x1:
                    part2 = self.Gs[imf, :, :].transpose()

                else:
                    part2 = self.Gs[imf, :, :].transpose().astype(np.uint8)
            if imf < self.Bs.shape[0]:
                if x1:
                    part3 = self.Bs[imf, :, :].transpose()
                else:
                    part3 = self.Bs[imf, :, :].transpose().astype(np.uint8)

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
            origin.imshow(self.img, cmap='gray', norm=NoNorm())
            origin.set_title('Original')
            decomp.imshow(self.ForShow(False), cmap='gray', norm=NoNorm())
            decomp.set_title('Reconstructed picture')

        else:
            fig, (origin, decomp) = plt.subplots(2, 1, figsize=(20, 20))
            origin.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            origin.set_title('Original')
            decomp.imshow(self.ForShow(False))
            decomp.set_title('Reconstructed picture')

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

        cv2.imwrite(curdir + 'ReConstructed.jpg', tmp)
        return True

    def show(self):
        x0 = fromarray(self.ForShow())
        x0.show()
