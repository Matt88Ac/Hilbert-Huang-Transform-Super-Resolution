from Develop.Interpolations import def_interpolations, imreadAndEMD, cv2
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
        interpolations = ['Gaussian', 'Bicubic', 'Bilinear', 'Lanczos5', 'Lanczos3', 'Lanczos4', 'MitchelCubic']

        for name in toOpen:
            image = cv2.imread(name, 0)
            rows, cols = image.shape
            new_image = cv2.resize(image, (int(cols / 6), int(rows / 6)), interpolation=cv2.INTER_LANCZOS4)
            decomposed = EMD2D(new_image)
            noIMfs = len(decomposed)

            upScaled = np.zeros(7)

            for i in range(7):
                upScaled[i] = self.__RMSE(image, def_interpolations(i)(new_image, (rows, cols)))

            new_one = np.zeros(image.shape)

            for i in range(len(decomposed)):
                data = [name, 'IMF ' + str(i+1), decomposed.MeanFrequency[i], decomposed.varFrequency[i],
                        rows, cols, decomposed.MedianFreq[i], decomposed.skewnessFreq[i], decomposed.kurtosisFreq[i],
                        decomposed.meanColor[i], decomposed.varColor[i], decomposed.medianColor[i],
                        decomposed.skewnessColor[i], decomposed.kurtosisColor[i]]

                interpolation = self.model.predict(data)







K = Run('SR_Results.csv')
