import numpy as np
from pyhht.emd import EmpiricalModeDecomposition as EMD
import cv2
from scipy import ndimage, signal, stats, interpolate
from matplotlib import pyplot as plt
from matplotlib.colors import NoNorm
from PIL.Image import fromarray, Image
from datetime import datetime
import os
from General_Scripts import Sharpen3x3, imread
from skimage.morphology import local_minima, local_maxima


class EMD2D:

    def __init__(self, image):
        self.IMFs: np.ndarray = np.array([])
        self.Rs = None
        self.Gs = None
        self.Bs = None
        self.NoIMFs: int = 0
        self.iter = 0
        self.EMD = EMD

        if image is None:
            return
        if type(image) == Image:
            self.img = np.array(image)

        else:
            self.img = image.copy()

        self.Error = np.zeros(self.img.shape, dtype=np.uint8)
        self.shape = self.img.shape

        self.__algorithm2()

        self.Error = self.img - self.reConstruct()
        self.varFrequency = np.zeros(self.__len__())
        self.MeanFrequency = np.zeros(self.__len__())
        self.skewnessFreq = np.zeros(self.__len__())
        self.kurtosisFreq = np.zeros(self.__len__())
        self.MedianFreq = np.zeros(self.__len__())
        self.entropyFreq = self.varFrequency.copy()
        self.shapiroFreq = self.varFrequency.copy()
        self.uniformityFreq = self.varFrequency.copy()

        self.meanColor = self.varFrequency.copy()
        self.varColor = self.meanColor.copy()
        self.skewnessColor = self.meanColor.copy()
        self.kurtosisColor = self.meanColor.copy()
        self.medianColor = self.varFrequency.copy()
        self.entropyColor = self.varFrequency.copy()
        self.shapiroColor = self.varFrequency.copy()
        self.uniformityColor = self.varFrequency.copy()

        for i in range(len(self)):
            if i < len(self):
                dtf = np.fft.fft2(self[i])
            else:
                dtf = np.fft.fft2(self.Error)

            r = (dtf.real ** 2 + dtf.imag ** 2) ** 0.5
            self.MeanFrequency[i] = np.mean(r)
            self.varFrequency[i] = np.var(r)
            self.skewnessFreq[i] = stats.moment(r, 3, axis=None)
            self.kurtosisFreq[i] = stats.moment(r, 4, axis=None)
            self.MedianFreq[i] = np.median(r)
            unique, counts = np.unique(r, return_counts=True, axis=None)
            counts = counts / counts.sum()
            self.entropyFreq[i] = stats.entropy(counts, axis=None)
            unique = stats.shapiro(r)
            self.shapiroFreq[i] = unique.statistic
            self.uniformityFreq[i] = np.sum(counts ** 2)

            self.meanColor[i] = self[i].mean()
            self.varColor[i] = self[i].var()
            self.skewnessColor[i] = stats.moment(self[i], 3, axis=None)
            self.kurtosisColor[i] = stats.moment(self[i], 4, axis=None)
            self.medianColor[i] = np.median(self[i])

            unique, counts = np.unique(self[i], return_counts=True, axis=None)
            counts = counts / counts.sum()
            self.entropyColor[i] = stats.entropy(counts, axis=None)

            unique = stats.shapiro(self[i])
            self.shapiroColor[i] = unique.statistic
            self.uniformityColor[i] = np.sum(counts ** 2)

        self.threshold = 0.5
        self.no_iterations = 15

    def __algorithm1(self):

        def getMinMax(matrix: np.ndarray):
            """
            Finds extrema, both mininma and maxima, based on local maximum filter.
            Returns extrema in form of two rows, where the first and second are
            positions of x and y, respectively.

            Parameters
            ----------
            image : numpy 2D array
                Monochromatic image or any 2D array.

            Returns
            -------
            min_peaks : numpy array
                Minima positions.
            max_peaks : numpy array
                Maxima positions.
            """

            # define an 3x3 neighborhood
            neighborhood = ndimage.generate_binary_structure(2, 2)

            # apply the local maximum filter; all pixel of maximal value
            # in their neighborhood are set to 1
            local_min = ndimage.maximum_filter(-matrix, footprint=neighborhood) == -matrix
            local_max = ndimage.maximum_filter(matrix, footprint=neighborhood) == matrix

            # can't distinguish between background zero and filter zero
            background = (matrix == 0)

            # appear along the bg border (artifact of the local max filter)
            eroded_background = ndimage.binary_erosion(background,
                                                       structure=neighborhood,
                                                       border_value=1)

            # we obtain the final mask, containing only peaks,
            # by removing the background from the local_max mask (xor operation)
            min_peaks = local_min ^ eroded_background
            max_peaks = local_max ^ eroded_background

            min_peaks = local_min
            max_peaks = local_max
            min_peaks[[0, -1], :] = False
            min_peaks[:, [0, -1]] = False
            max_peaks[[0, -1], :] = False
            max_peaks[:, [0, -1]] = False

            min_peaks = np.nonzero(min_peaks)
            max_peaks = np.nonzero(max_peaks)

            return min_peaks, max_peaks

        def getSplines1(matrix: np.ndarray):

            def getMin():
                fp = np.ones((3, 3))

                ind_min = local_minima(image=matrix, selem=fp, connectivity=False, allow_borders=False, indices=True)
                val_min = local_minima(image=matrix, selem=fp, connectivity=False, allow_borders=False)
                minSpline = interpolate.Rbf(ind_min[0], ind_min[1], matrix[val_min], function='thin_plate')

                xi = np.array(range(matrix.shape[0]))
                yi = np.array(range(matrix.shape[1]))
                xi, yi = np.meshgrid(xi, yi)

                return minSpline(xi, yi).transpose()

            def getMax():
                fp = np.ones((3, 3))

                ind_max = local_maxima(image=matrix, selem=fp, connectivity=False, allow_borders=False, indices=True)
                val_max = local_maxima(image=matrix, selem=fp, connectivity=False, allow_borders=False)
                maxSpline = interpolate.Rbf(ind_max[0], ind_max[1], matrix[val_max], function='thin_plate')

                xi = np.array(range(matrix.shape[0]))
                yi = np.array(range(matrix.shape[1]))
                xi, yi = np.meshgrid(xi, yi)

                return maxSpline(xi, yi).transpose()

            return getMin(), getMax()

        def getSplines2(matrix: np.ndarray):
            mins, maxs = getMinMax(matrix)
            minsVal = matrix[mins]
            maxsVal = matrix[maxs]

            def getUpper():
                maxSpline = interpolate.Rbf(maxs[0], maxs[1], maxsVal, function='thin_plate')
                xi = np.array(range(matrix.shape[0]))
                yi = np.array(range(matrix.shape[1]))
                xi, yi = np.meshgrid(xi, yi)

                return maxSpline(xi, yi).transpose()

            def getLower():
                minSpline = interpolate.Rbf(mins[0], mins[1], minsVal, function='thin_plate')
                xi = np.array(range(matrix.shape[0]))
                yi = np.array(range(matrix.shape[1]))
                xi, yi = np.meshgrid(xi, yi)

                return minSpline(xi, yi).transpose()

            return getLower(), getUpper()

        def Check_IMF(candidate: np.ndarray, prev: np.ndarray, mean: np.ndarray):

            if np.mean(np.abs(candidate)) < 0.5:
                return True

            if np.allclose(candidate, prev, 0.5):
                return True

            if np.all(np.abs(mean - mean.mean()) < 0.5):
                return True

            if np.all(np.abs(mean) < 0.5):
                return True

            return False

        def Sift(matrix: np.ndarray):
            lower, upper = getSplines2(matrix)
            mean = (lower + upper) / 2

            new_imf = matrix.copy() - mean
            prev = matrix.copy()
            i = 0
            while not Check_IMF(new_imf, prev, mean):
                if i == 15:
                    print('got to limit')
                    break
                prev = new_imf.copy()
                lower, upper = getSplines2(new_imf)
                mean = (lower + upper) / 2
                new_imf = new_imf - mean
                i += 1

            return new_imf

        def Run(img: np.ndarray) -> np.ndarray:
            self.IMFs = Sift(img)
            self.IMFs = self.IMFs.reshape((1, self.shape[0], self.shape[1]))
            i = 1
            self.NoIMFs = 1
            while not np.allclose(self.IMFs[i - 1], 0):
                self.save()
                if i == 1:
                    temp_IMF = Sift(img - self.IMFs[0])
                    temp_IMF = temp_IMF.reshape((1, self.shape[0], self.shape[1]))
                else:
                    temp_IMF = Sift(self.IMFs[i - 2] - self.IMFs[i - 1])
                    temp_IMF = temp_IMF.reshape((1, self.shape[0], self.shape[1]))

                self.IMFs = np.concatenate((self.IMFs, temp_IMF), axis=0)
                i += 1
                self.NoIMFs += 1
            self.save()
            return self.IMFs.copy()

        if len(self.img.shape) == 3:
            self.Rs = Run(self.img[:, :, 0])
            self.Gs = Run(self.img[:, :, 1])
            self.Bs = Run(self.img[:, :, 2])
            self.NoIMFs = max(self.Rs.shape[0], self.Bs.shape[0], self.Gs.shape[0]) + 1

        else:
            Run(self.img)
            self.NoIMFs += 1

    def __algorithm2(self):
        def Run(img: np.ndarray):
            self.IMFs = np.array([])

            def emd_images_col(colOfImage: np.ndarray):
                to_ret = self.EMD(colOfImage).decompose()
                return to_ret

            for i in range(img.shape[1]):
                newImf = emd_images_col(img[:, i])
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

            return self.IMFs

        if len(self.shape) == 2:
            Run(self.img)
            self.NoIMFs += 1

        else:
            self.Rs = Run(self.img[:, :, 0])
            self.Gs = Run(self.img[:, :, 1])
            self.Bs = Run(self.img[:, :, 2])

            self.NoIMFs = max(self.Rs.shape[0], self.Bs.shape[0], self.Gs.shape[0]) + 1

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
                tmp[ln] = self.__call(i, dtype)
                ln += 1
            return tmp

        else:
            if imf == len(self) + 1:
                return self.Error

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
                return np.sum(self.IMFs, axis=0).transpose() + self.Error

            R = np.sum(self.Rs, axis=0).transpose()
            G = np.sum(self.Gs, axis=0).transpose()
            B = np.sum(self.Bs, axis=0).transpose()

            return cv2.merge((R, G, B)) + self.Error

        else:
            if len(self.shape) == 2:
                return np.sum(self.IMFs, axis=0).transpose().astype(dtype) + self.Error

            R = np.sum(self.Rs, axis=0).transpose().astype(dtype)
            G = np.sum(self.Gs, axis=0).transpose().astype(dtype)
            B = np.sum(self.Bs, axis=0).transpose().astype(dtype)

            return cv2.merge((R, G, B)) + self.Error

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

        if type(self.MeanFrequency) == tuple:
            tmp.MeanFrequency = self.MeanFrequency
            tmp.stdFrequency = self.varFrequency
        else:
            tmp.MeanFrequency = self.MeanFrequency.copy()
            tmp.stdFrequency = self.varFrequency.copy()

        tmp.NoIMFs = self.NoIMFs
        return tmp

    """""""""
    def __repr__(self):
        tmp = self.ForShow(median_filter=False)
        if len(self.shape) == 2:
            plt.imshow(tmp, cmap='gray', norm=NoNorm())
        else:
            plt.imshow(tmp)
        plt.show()
        return ""
    """

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
        """""""""
        np.save(curdir + 'mean_frequency.npy', self.MeanFrequency)
        np.save(curdir + 'var_frequency.npy', self.varFrequency)
        if len(self.shape) == 2:
            np.save(curdir + 'IMF_array.npy', self.IMFs)
        else:
            np.save(curdir + 'IMF_R.npy', self.Rs)
            np.save(curdir + 'IMF_G.npy', self.Gs)
            np.save(curdir + 'IMF_B.npy', self.Bs)
        np.save(curdir + 'IMF_Error.npy', self.Error)
        """

        if with_imfs:
            for i in range(len(self)):
                tmp1 = self.__getitem__(i)
                cv2.imwrite(curdir + 'IMF_' + str(i + 1) + '.jpg', tmp1)

        cv2.imwrite(curdir + 'ReConstructed.jpg', tmp)
        return True

    def show(self, which=None):
        if not which:
            x0 = fromarray(self.ForShow())
            x0.show()
        else:
            x0 = self(which)
            if len(self.shape) == 2:
                x0 = fromarray(x0)
            else:
                x0 = cv2.cvtColor(x0, cv2.COLOR_BGR2RGB)
                x0 = fromarray(x0)
            x0.show()
