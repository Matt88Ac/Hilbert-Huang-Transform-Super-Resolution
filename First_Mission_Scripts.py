from PyEMD.EMD2d import EMD2D, print_function, maximum_filter
from PyEMD.EMD import EMD
from PyEMD.EEMD import EEMD
import numpy as np
import pandas as pd
from matplotlib.pyplot import imshow, imread
import cv2
import os
from PIL.Image import fromarray


class Run:
    def __init__(self, csv_name: str):
        self.table = pd.read_csv(csv_name)
        self.emd = EMD2D()
        self.name = csv_name

    def checkExistence(self):
        temp = self.table['File Name'].copy()
        temp = np.array(temp, dtype=str)
        l1 = self.getFileNames()
        if len(temp) == 0:
            return l1
        return l1[l1 not in temp]

    def getFileNames(self):
        dirc = os.getcwd()
        dirc = dirc.replace(dirc[2], '/') + '/DATA'
        self.dir = dirc
        return np.array(os.listdir(dirc))

    def AddToCSV(self, fname: str, mode, resolution, mean, median, mx, mn, imfs, rmse):
        to_append = pd.DataFrame({'File Name': [fname],
                                  'Color Mode': [mode],
                                  'Resolution': [resolution],
                                  'Mean Pixel Value': [mean],
                                  'Median Pixel Value': [median],
                                  'Max Pixel Value': [mx],
                                  'Min Pixel Value': [mn],
                                  'Number of IMFs': [imfs],
                                  'RMSE': rmse})
        self.table = self.table.append(to_append)
        self.table.to_csv(self.name, index=False)

    def RunGreys(self):
        toOpen = self.checkExistence()

        def RMSE(expected: np.ndarray, estimated: np.ndarray):
            diff = np.sum((expected - estimated) ** 2)
            pixSize = expected.shape[0] * expected.shape[1]
            return (diff / pixSize) ** 0.5

        for name in toOpen:
            fname = 'DATA/' + name
            img = cv2.imread(fname, 0)
            resolution = img.shape
            color_mode = 'Grey'

            maxp = img.max()
            minp = img.min()
            med = np.median(img)
            mean = img.mean()
            try:
                picDecomposed = self.emd.emd(img)
            finally:
                print('ALALA')
                os.remove(self.dir + name)

            numOfIMFs = picDecomposed.shape[0]
            rmse = RMSE(img, np.sum(picDecomposed, axis=0))

            self.AddToCSV(fname, color_mode, resolution, mean, med, maxp, minp, numOfIMFs, rmse)


x = Run('FirstDataFrame1.csv')
x.RunGreys()
