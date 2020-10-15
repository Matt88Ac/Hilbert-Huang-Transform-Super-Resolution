from Develop.Interpolations import def_interpolations, cv2
from Develop.EMD2D import EMD2D
import pandas as pd
import os
import numpy as np
import pickle
import joblib


class Run:
    def __init__(self, name: str):
        self.dir = None
        self.name = name
        self.table = pd.read_csv(name, index_col=False)

        names = pd.read_csv('interpolations.csv')
        names = names['File Name'].unique()

        self.files = names
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

    def AddToCSV(self, NoIMF, name, resolution, HHT2D, Bicubic, Lanczos, Gaussian, Bilinear, Best):
        rows = resolution[0]
        cols = resolution[1]
        to_append = pd.DataFrame({'File Name': [name],
                                  'No IMFs': [NoIMF],
                                  'RMSE - HHT2D': [HHT2D],
                                  'RMSE - Bicubic': [Bicubic],
                                  'RMSE - Lanczos': [Lanczos],
                                  'RMSE - Gaussian': [Gaussian],
                                  'RMSE - Bilinear': [Bilinear],
                                  'Best Interpolation Method': [Best],
                                  'No Rows': [rows],
                                  'No Cols': [cols],
                                  })
        self.table = self.table.append(to_append)
        self.table.to_csv(self.name, index=False)

    def __RMSE(self, expected: np.ndarray, estimated: np.ndarray):
        return ((expected - estimated) ** 2).mean() ** 0.5

    def runner(self):
        toOpen = self.checkExistence()
        interpolations = np.array(['Gaussian', 'Bicubic', 'Bilinear', 'Lanczos5', 'Lanczos3', 'Lanczos4', 'MitchelCubic'])

        for name in toOpen:
            image = cv2.imread('DATA/' + name, 0)
            print(name)
            rows, cols = image.shape
            new_image = cv2.resize(image, (int(cols / 6), int(rows / 6)), interpolation=cv2.INTER_LANCZOS4)
            decomposed = EMD2D(new_image)
            noIMfs = len(decomposed)

            upScaled = np.zeros(7)

            new_image = new_image.reshape((new_image.shape[0], new_image.shape[1], 1))
            for i in range(7):
                temp = def_interpolations[i](new_image, (rows, cols))
                if len(temp.shape) == 3:
                    upScaled[i] = self.__RMSE(image, temp[:, :, 0])
                else:
                    upScaled[i] = self.__RMSE(image, temp)

            new_one = np.zeros(image.shape)

            for i in range(len(decomposed)):

                data = [[0, decomposed.MeanFrequency[i], decomposed.varFrequency[i],
                        rows, cols, decomposed.MedianFreq[i], decomposed.skewnessFreq[i], decomposed.kurtosisFreq[i],
                        decomposed.meanColor[i], decomposed.varColor[i], decomposed.medianColor[i],
                        decomposed.skewnessColor[i], decomposed.kurtosisColor[i]]]

                interpolation = self.model.predict(data)

                for j in range(7):
                    if interpolation == interpolations[j]:
                        temp = def_interpolations[j](decomposed(i).reshape((decomposed.shape[0], decomposed.shape[1], 1)), (rows, cols))
                        if len(temp.shape) == 3:
                            temp = temp[:, :, 0]
                        new_one += temp
                        break

            hht_rmse = self.__RMSE(image, new_one)

            best = min(upScaled.min(), hht_rmse)

            if best == hht_rmse:
                best = 'HHT'
            else:
                best = interpolations[upScaled == best]

            self.AddToCSV(NoIMF=noIMfs, name=name, resolution=image.shape, Bicubic=upScaled[1], Gaussian=upScaled[0],
                          Bilinear=upScaled[2],  Best=best, Lanczos=upScaled[5], HHT2D=hht_rmse)



K = Run('SR_Results.csv')
