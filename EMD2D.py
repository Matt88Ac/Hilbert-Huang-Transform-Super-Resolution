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


class EMD2D:
    Rs = None
    Gs = None
    Bs = None
    NoIMFs: int = 0
    IMFs = []

    def __init__(self, image: np.ndarray):
        if image is None:
            return

        self.EMD = EMD
        self.img = image

        self.shape = image.shape

        def emd_images_col(colOfImage: np.ndarray) -> np.ndarray:
            return self.EMD(colOfImage).decompose()

        def AddDecomposed(newIMF: np.ndarray):
            # newArr: np.ndarray = np.vstack((self.IMFs[0], newIMF[0]))
            ln = min(self.NoIMFs, newIMF.shape[0])
            k = 0
            for i in range(ln):
                self.IMFs[i] = np.vstack((self.IMFs[i], newIMF[i]))
                k += 1

            if self.NoIMFs == newIMF.shape[0]:
                return

            elif ln == newIMF.shape[0]:
                return

            for i in range(k, newIMF.shape[0]):
                self.IMFs.append(newIMF[i].copy())

            self.NoIMFs = newIMF.shape[0]

        def Run(img: np.ndarray):
            deco = emd_images_col(img[:, 0])
            self.NoIMFs = deco.shape[0]

            for i in range(len(deco)):
                self.IMFs.append(deco[i])

            for i in range(1, img.shape[1]):
                deco = emd_images_col(img[:, i])
                AddDecomposed(deco.copy())

            maxim = self.shape[1]
            newArr = np.empty((self.NoIMFs, self.shape[1], self.shape[0]))
            print(newArr.shape)
            print(self.IMFs[0].shape)

            for i in range(len(self.IMFs)):
                howMuch = maxim - self.IMFs[i].shape[0]
                if len(self.IMFs[i].shape) == 1:
                    continue

                if howMuch == 0:
                    newArr[i] = self.IMFs[i].copy()
                    continue

                concat = np.zeros((howMuch, self.shape[0]))
                newArr[i] = np.vstack((self.IMFs[i], concat))

            self.IMFs = newArr.copy()
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

    def surfaces(self):
        return
        tmp = self.ForShow(median_filter=False)
        x0, y0 = tmp.shape
        x0, y0 = np.meshgrid(range(y0), range(x0))
        if len(self.shape) == 2:
            fig = plt.figure(figsize=(20, 20))
            ax = Axes3D(fig=fig)
            # ax = plt.axes(projection='3d')
            ax.plot_surface(x0, y0, self.img - tmp, cmap='viridis')
            ax.grid()
            ax.set_zlim(0, 255)
            # ax.plot_wireframe(x0, y0, tmp, color='black')
            plt.show()
            return ax


        else:
            fig = plt.figure(figsize=(20, 20))
            ax = Axes3D(fig=fig)
            ax.plot_surface(x0, y0, self.img[:, :, 0], color='red')
            ax.grid()
            ax.set_zlim(0, 255)
            ax.plot_surface(x0, y0, tmp[:, :, 0], color='black')

            ax = fig.add_subplot(1, 2, 2, projection='3d')
            ax.plot_surface(x0, y0, self.img[:, :, 1], color='green')
            ax.grid()
            ax.set_zlim(0, 255)
            ax.plot_surface(x0, y0, tmp[:, :, 1], color='black')
            ax = fig.add_subplot(1, 2, 3, projection='3d')
            ax.plot_surface(x0, y0, self.img[:, :, 2], color='blue')
            ax.grid()
            ax.set_zlim(0, 255)
            ax.plot_surface(x0, y0, tmp[:, :, 2], color='black')
            plt.show()


img = cv2.imread('DATA/dog.jpg', 0)
dec = EMD2D(img)
x1 = fromarray(dec.ForShow(False))
x1.show()
