import numpy as np
from pyhht.emd import EmpiricalModeDecomposition as EMD
import cv2
from scipy import ndimage, signal
import scipy.fft as fft
from matplotlib import pyplot as plt
from matplotlib.colors import NoNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL.Image import fromarray
from datetime import datetime
import os

Sharpen3x3 = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
Sharpen3x3 = Sharpen3x3.reshape((3, 3))


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
        if len(self.shape) == 3:
            self.MeanFrequency = np.array([[], [], []])
        else:
            self.MeanFrequency = np.array([])
        self.stdFrequency = self.MeanFrequency.copy()

        def Run(img: np.ndarray, n=0):

            def emd_images_col(colOfImage: np.ndarray):
                to_ret = self.EMD(colOfImage).decompose()
                return to_ret  # .reshape((to_ret.shape[1], to_ret.shape[0]))

            def fftMyIMF(imf: np.ndarray) -> np.ndarray:
                return np.fft.fft(imf).real

            def decAndFFT(colOfImage: np.ndarray) -> np.ndarray:
                decCol = emd_images_col(colOfImage)
                dft = fftMyIMF(decCol)

                if len(self.MeanFrequency.shape) == 1:
                    self.MeanFrequency = np.append(self.MeanFrequency, dft.mean(axis=1))
                    self.stdFrequency = np.append(self.stdFrequency, dft.std(axis=1))

                else:
                    self.MeanFrequency[n] = np.append(self.MeanFrequency[n], dft.mean(axis=1))
                    self.stdFrequency[n] = np.append(self.stdFrequency[n], dft.std(axis=1))

                return decCol

            temp_imfs = []
            size = 0
            max_size = 0
            for i in range(img.shape[1]):
                dec1 = decAndFFT(img[:, i])
                tnum = dec1.shape[0]
                temp_imfs.append(np.array2string(dec1, separator=',', suppress_small=False))
                if tnum > max_size:
                    max_size = tnum
                size += 1
            self.NoIMFs = max_size
            temp_imfs = np.array(temp_imfs)
            dec1 = None

            if len(self.shape) == 2:
                mx = np.max(self.MeanFrequency)
                mn = np.min(self.MeanFrequency)

            else:
                mx = self.MeanFrequency[n].max()
                mn = self.MeanFrequency[n].min()

            indicator = np.empty((max_size, 2))
            diffs = (mx - mn) / max_size
            indicator[0] = np.array([mn, mn + diffs])
            for i in range(1, max_size):
                indicator[i] = np.array([indicator[i - 1][1], diffs + indicator[i - 1][1]])

            def classifySpots(ranges: np.ndarray, values: np.ndarray):
                tf = 0
                for ind in range(len(values)):
                    t1 = ranges[:, 0] <= values[ind]
                    t2 = ranges[:, 1] > values[ind]
                    if ind == 0:
                        tf = t1 * t2
                    else:
                        tt = t1 * t2
                        if tf[tt == True]:
                            nk = 0
                            tt = tt == True
                            tt = np.arange(1, len(ranges) + 1) * tt.astype(int)
                            tt = tt[tt != 0]
                            tt -= 1
                            while True:
                                tt = tt % len(ranges)
                                if not tf[tt]:
                                    tf[tt] = True
                                    break
                                tt += 1
                        else:
                            tf += tt

                to_ret = tf.astype(int) * np.arange(1, len(ranges) + 1)
                to_ret = to_ret[to_ret != 0]
                return tf

            def deFrozeIMFs(imfs: str) -> np.ndarray:
                temp = imfs[1:len(imfs) - 1]
                temp = temp.split('],')
                arr = np.empty((len(temp), self.shape[0]))
                for j in range(len(temp)):
                    t1 = temp[j].replace('\n', '').replace('[', '')
                    if t1 == '':
                        continue
                    arr[j] = np.fromstring(t1, sep=',', dtype=float)
                return arr.transpose()

            k = 0
            for i in range(len(temp_imfs)):
                dec = deFrozeIMFs(temp_imfs[i])
                dec = dec.reshape((1, dec.shape[0], dec.shape[1]))
                if dec.shape[2] == max_size:
                    k += dec.shape[2]
                    if len(self.IMFs) == 0:
                        self.IMFs = dec.copy()
                        # self.IMFs = self.IMFs.reshape((1, self.IMFs.shape[0], self.IMFs.shape[1]))
                        continue
                    else:
                        self.IMFs = np.concatenate((self.IMFs, dec), axis=0)
                        # newArr = np.array([np.vstack((self.IMFs[0], dec[0]))])
                        # for j in range(1, max_size):
                        #    newArr = np.vstack((newArr, np.array([np.vstack((self.IMFs[i], dec[i]))])))
                        # self.IMFs = newArr.copy()
                        continue
                else:
                    if len(self.shape) == 2:
                        rel_mean = self.MeanFrequency[k: k + dec.shape[2]]
                    else:
                        rel_mean = self.MeanFrequency[n][k: k + dec.shape[2]]
                    spots = classifySpots(indicator, rel_mean)
                    toAdd = np.zeros((1, dec.shape[1], max_size))
                    # toAdd[0, :, spots] = dec[0, :, :]
                    for kk in range(len(spots)):
                        if spots[kk]:
                            toAdd[0, :, kk] = dec[0, :, kk]

                    if len(self.IMFs) == 0:
                        self.IMFs = toAdd.copy()

                    else:
                        self.IMFs = np.concatenate((self.IMFs, toAdd), axis=0)
            return self.IMFs

        if len(self.shape) == 2:
            Run(image)
            errorImf = self.img - self.reConstruct()
            errorImf = errorImf.reshape((1, errorImf.shape[0], errorImf.shape[1]))
            self.IMFs = np.concatenate((self.IMFs, errorImf), axis=2)
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
        def act(Imfs: np.ndarray, axis=2):
            return np.sum(Imfs, axis=axis)

        if len(self.shape) == 2:
            return act(self.IMFs).astype(np.uint8)

        return cv2.merge((act(self.Rs).transpose().astype(np.uint8), act(self.Gs).transpose().astype(np.uint8),
                          act(self.Bs).transpose().astype(np.uint8)))

    def ForShow(self, median_filter=False):
        if len(self.shape) == 2:
            ret = self.reConstruct()
            if median_filter:
                ret = ndimage.median_filter(ret, 3)
            return ret

        ret = cv2.cvtColor(self.reConstruct(), cv2.COLOR_BGR2RGB)
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

        if not sigma:
            sigma = 0.1
        if gaussian:
            temp = ndimage.gaussian_filter(temp, sigma)

        if laplace:
            temp = ndimage.laplace(temp)

        if LoG:
            temp = ndimage.gaussian_laplace(temp, sigma)

        if median:
            temp = ndimage.median_filter(temp, 3)

        if sobel:
            temp = ndimage.sobel(temp)

        if prewitt:
            temp = ndimage.prewitt(temp)

        if sharp:
            if len(temp.shape) == 2:
                return signal.convolve2d(temp, Sharpen3x3, mode='same')
            temp[:, :, 0] = signal.convolve2d(temp[:, :, 0], Sharpen3x3, mode='same')
            temp[:, :, 1] = signal.convolve2d(temp[:, :, 1], Sharpen3x3, mode='same')
            temp[:, :, 2] = signal.convolve2d(temp[:, :, 2], Sharpen3x3, mode='same')

        return temp

    def applyFFT(self, median_filter=False, as_int=False):
        dx = self.ForShow(median_filter=median_filter)
        f1 = fft.fft(dx)

        return f1.real * (1 - int(as_int)) + f1.real.astype(np.uint8) * int(as_int), f1

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


im = cv2.imread('DATA/dog.jpg', 0)
deco = EMD2D(im)
