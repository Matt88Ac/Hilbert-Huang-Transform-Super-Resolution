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

    def AddToCSV(self, NoIMF, name, resolution, rmse, psnr, ssim, hht_rmse, hht_psnr, hht_ssim, from_row, from_col):
        rows = resolution[0]
        cols = resolution[1]
        to_append = pd.DataFrame({'File Name': [name],
                                  'No IMFs': [NoIMF],
                                  'No Rows': [rows],
                                  'No Cols': [cols],
                                  'xRows': [from_row],
                                  'xCols': [from_col],
                                  'Best RMSE': [rmse[0]],
                                  'Best SSIM': [ssim[0]],
                                  'Best PSNR': [psnr[0]],
                                  'HHT RMSE': [hht_rmse],
                                  'HHT SSIM': [hht_ssim],
                                  'HHT PSNR': [hht_psnr],
                                  'Best RMSE - Value': [rmse[1]],
                                  'Best SSIM - Value': [ssim[1]],
                                 'Best PSNR - Value': [psnr[1]]
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
            if name == '00903.jpg':
                continue
            image = cv2.imread('DATA/' + name, 0)
            print(name)
            rows, cols = image.shape
            new_row = abs(np.random.normal(1.5, 5, 1)[0]) + 1
            new_col = abs(np.random.normal(1.5, 5, 1)[0]) + 1
            new_image = cv2.resize(image, (int(cols / new_col), int(rows / new_row)), interpolation=cv2.INTER_LANCZOS4)
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
                            decomposed[i].reshape((decomposed.shape[0], decomposed.shape[1], 1)), (rows, cols))
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

            b1 = max(ims_psnr.max(), hht_psnr)
            b2 = max(ims_ssim.max(), hht_ssim)
            b3 = min(ims_rmse.min(), hht_rmse)
            v1 = b1
            v2 = b2
            v3 = b3

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

            self.AddToCSV(NoIMF=noIMfs, name=name, resolution=image.shape, rmse=(b3, v3), psnr=(b1, v1),
                          ssim=(b2, v2), hht_psnr=hht_psnr, hht_ssim=hht_ssim, hht_rmse=hht_rmse, from_col=new_col,
                          from_row=new_row)


K = Run('SR_Results.csv')
