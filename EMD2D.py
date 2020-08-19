import numpy as np
from pyhht.emd import EmpiricalModeDecomposition as EMD
from PyEMD.EEMD import EEMD
import cv2
from PIL.Image import fromarray


class EMD2D:
    IMFs: np.ndarray = np.array([])
    NoIMFs: int = 0

    def __init__(self, image: np.ndarray):
        self.EEMD = EEMD()
        self.EMD = EMD

        def EEMD_images_col(colOfImage: np.ndarray):
            return self.EEMD.eemd(colOfImage)

        def emd_images_col(colOfImage: np.ndarray):
            return self.EMD(colOfImage).decompose()

        def concatZeros(row: np.ndarray, howMuch: int, bo) -> np.ndarray:
            if not bo:
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

        def Run(img: np.ndarray):
            deco = emd_images_col(img[:, 0])
            self.NoIMFs = deco.shape[0]
            self.IMFs = deco.copy()
            # for i in range(1, deco.shape[0]):
            #   self.IMFs = np.vstack((self.IMFs, np.array([deco[i]])))

            for i in range(1, img.shape[1]):
                deco = emd_images_col(img[:, i])
                AddDecomposed(deco.copy())

        if len(image.shape) == 3:
            Run(image[:, :, 0])
            Run(image[:, :, 1])
            Run(image[:, :, 2])

        else:
            Run(image)


imge = cv2.imread('DATA/input.jpg', 0)
x = EMD2D(image=imge)
print(x.NoIMFs)