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
