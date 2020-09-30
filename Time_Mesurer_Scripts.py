import platform
from Develop.EMD2D import EMD2D
import numpy as np
import pandas as pd
import cv2
import os
import time


class Run:
    def __init__(self, csv_name: str):
        self.table = pd.read_csv(csv_name, index_col=False)
        self.emd = EMD2D
        self.name = csv_name
        self.platform = platform.system()
        if self.platform == 'Windows':
            dirc = os.getcwd()
            dirc = dirc.replace(dirc[2], '/') + '/DATA'
            self.dir = dirc
        else:
            dirc = os.getcwd() + "/DATA"
            self.dir = dirc

        self.RunGreys()

    def checkExistence(self):
        temp = self.table['File Name'].copy()
        temp = np.array(temp, dtype=str)
        l1 = self.getFileNames()
        if len(temp) == 0:
            return l1
        l2 = ['DATA/' + y for y in l1]
        l1 = np.setdiff1d(l2, temp)
        return np.array([y[5:] for y in l1], dtype=str)

    def getFileNames(self):
        return np.array(os.listdir(self.dir), dtype=str)

    def AddToCSV(self, fname: str, resolution, imfs, rmse, timer):
        to_append = pd.DataFrame({'File Name': [fname],
                                  'Shape': [resolution],
                                  'No IMFs': [imfs],
                                  'Time': [timer],
                                  'RMSE': [rmse]
                                  })
        self.table = self.table.append(to_append)
        self.table.to_csv(self.name, index=False)

    def RunGreys(self):
        toOpen = self.checkExistence()

        def RMSE(expected: np.ndarray, estimated: np.ndarray):
            diff = np.sum(((expected - estimated) ** 2).mean()) ** 0.5
            return diff

        for name in toOpen:
            print(name)
            fname = 'DATA/' + name  # 'road_image.jpg'
            img = cv2.imread(fname, 0)
            resolution = img.shape
            try:
                start = time.clock()
                picDecomposed = self.emd(img)
                end = time.clock()
                picDecomposed.save()
                numOfIMFs = picDecomposed.IMFs.shape[0]
                rmse = RMSE(img, picDecomposed.reConstruct())

                self.AddToCSV(fname=fname, resolution=resolution, rmse=rmse, imfs=numOfIMFs, timer=end - start)
            except ValueError:
                print("Error occured during process {}".format(name))


cos = Run('Time_Mesure.csv')
