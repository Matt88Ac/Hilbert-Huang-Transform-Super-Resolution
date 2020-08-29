import numpy as np
from pyhht.emd import EmpiricalModeDecomposition as EMD
import cv2


class EMD2D:
    IMFs: np.ndarray = np.array([])
    Rs = None
    Gs = None
    Bs = None
    NoIMFs: int = 0

    def __init__(self, image: np.ndarray):
        self.EMD = EMD

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
            return self.IMFs[imf].transpose().astype(np.uint8)

        part1 = part2 = part3 = np.zeros((self.shape[0], self.shape[1]))

        if imf < self.Rs.shape[2]:
            part1 = self.Rs[imf, :, :].transpose().astype(np.uint8)

        if imf < self.Gs.shape[2]:
            part2 = self.Gs[imf, :, :].transpose().astype(np.uint8)

        if imf < self.Bs.shape[2]:
            part3 = self.Bs[imf, :, :].transpose().astype(np.uint8)

        return cv2.merge((part1, part2, part3))


    def reConstruct(self):
        def act(Imfs: np.ndarray, axis=0):
            return np.sum(Imfs, axis=axis)

        if len(self.shape) == 2:
            return act(self.IMFs).transpose().astype(np.uint8)

        return cv2.merge((act(self.Rs).transpose().astype(np.uint8), act(self.Gs).transpose().astype(np.uint8),
                          act(self.Bs).transpose().astype(np.uint8)))
