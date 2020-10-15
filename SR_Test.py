from Develop.Interpolations import def_interpolations, cv2
from Develop.EMD2D import EMD2D
import pandas as pd
import os
import numpy as np
import pickle
from Develop.SRMetrices import PSNR, SSIM, Normalized_RMSE


class Run:
    def __init__(self, name: str):
        self.dir = None
        self.name = name
        self.table = pd.read_csv(name, index_col=False)

        names = pd.read_csv('interpolations.csv')
        names = names['File Name'].unique()

        self.files = np.append(names, self.table['File Name'])
        self.model = pickle.load(open('random_forest_model.pkl', 'rb'))

        self.runner()

    def checkExistence(self):
        temp = self.files
        temp = np.array(temp, dtype=str)
        l1 = self.getFileNames()
        if len(temp) == 0:
            return l1
        l1 = np.setdiff1d(l1, temp)
        return np.array(l1, dtype=str)

    def getFileNames(self):
        dirc = os.getcwd()
        dirc = dirc.replace(dirc[2], '/') + '/DATA'
        self.dir = dirc
        return np.array(os.listdir(dirc), dtype=str)

    def AddToCSV(self, NoIMF, name, resolution, rmse, psnr, ssim):
        rows = resolution[0]
        cols = resolution[1]
        to_append = pd.DataFrame({'File Name': [name],
                                  'No IMFs': [NoIMF],
                                  'No Rows': [rows],
                                  'No Cols': [cols],
                                  'Best RMSE': [rmse],
                                  'Best SSIM': [ssim],
                                  'Best PSNR': [psnr]
                                  })
        self.table = self.table.append(to_append)
        self.table.to_csv(self.name, index=False)

    def __RMSE(self, expected: np.ndarray, estimated: np.ndarray):
        return ((expected - estimated) ** 2).mean() ** 0.5

    def runner(self):
        toOpen = self.checkExistence()
        interpolations = np.array(
            ['Gaussian', 'Bicubic', 'Bilinear', 'Lanczos5', 'Lanczos3', 'Lanczos4', 'MitchelCubic'])

        for name in toOpen:
            image = cv2.imread('DATA/' + name, 0)
            print(name)
            rows, cols = image.shape
            new_image = cv2.resize(image, (int(cols / 6), int(rows / 6)), interpolation=cv2.INTER_LANCZOS4)
            decomposed = EMD2D(new_image)
            noIMfs = len(decomposed)
            print('done EMD')

            upScaled = np.zeros((7, rows, cols))

            new_image = new_image.reshape((new_image.shape[0], new_image.shape[1], 1))
            for i in range(7):
                temp = def_interpolations[i](new_image, (rows, cols))
                if len(temp.shape) == 3:
                    temp = temp[:, :, 0]
                upScaled[i] = temp.copy()

            new_one = np.zeros(image.shape)

            for i in range(len(decomposed)):

                data = [[0, decomposed.MeanFrequency[i], decomposed.varFrequency[i],
                         rows, cols, decomposed.MedianFreq[i], decomposed.skewnessFreq[i], decomposed.kurtosisFreq[i],
                         decomposed.meanColor[i], decomposed.varColor[i], decomposed.medianColor[i],
                         decomposed.skewnessColor[i], decomposed.kurtosisColor[i]]]

                interpolation = self.model.predict(data)

                for j in range(7):
                    if interpolation == interpolations[j]:
                        temp = def_interpolations[j](
                            decomposed(i).reshape((decomposed.shape[0], decomposed.shape[1], 1)), (rows, cols))
                        if len(temp.shape) == 3:
                            temp = temp[:, :, 0]
                        new_one += temp
                        break

            hht_rmse = Normalized_RMSE(image, new_one)
            hht_psnr = abs(PSNR(image, new_one))
            hht_ssim = abs(SSIM(image, new_one))

            ims_psnr = np.array([abs(PSNR(image, upScaled[i])) for i in range(7)])
            ims_ssim = np.array([abs(SSIM(image, upScaled[i])) for i in range(7)])
            ims_rmse = np.array([Normalized_RMSE(image, upScaled[i]) for i in range(7)])

            b1 = min(ims_psnr.min(), hht_psnr)
            b2 = min(ims_ssim.min(), hht_ssim)
            b3 = min(ims_rmse.min(), hht_rmse)

            if b1 == hht_psnr:
                b1 = 'HHT'
            else:
                b1 = interpolations[b1 == ims_psnr][0]

            if b2 == hht_ssim:
                b2 = 'HHT'
            else:
                b2 = interpolations[ims_ssim == b2][0]

            if b3 == hht_rmse:
                b3 = 'HHT'
            else:
                b3 = interpolations[ims_rmse == b3][0]

            self.AddToCSV(NoIMF=noIMfs, name=name, resolution=image.shape, rmse=b3, psnr=b1, ssim=b2)


K = Run('SR_Results.csv')
