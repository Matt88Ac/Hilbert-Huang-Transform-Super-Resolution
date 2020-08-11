from PyEMD.EMD2d import EMD2D
import numpy as np
import pandas as pd
from matplotlib.pyplot import imshow, imread
import cv2
import os
from PIL.Image import fromarray
import time


class Run:
    def __init__(self, csv_name: str):
        self.table = pd.read_csv(csv_name)
        self.table = self.table.dropna()
        self.table = self.table.drop_duplicates(subset='File Name', keep=False)
        self.emd = EMD2D()
        self.name = csv_name

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
        dirc = os.getcwd()
        dirc = dirc.replace(dirc[2], '/') + '/DATA'
        self.dir = dirc
        return np.array(os.listdir(dirc), dtype=str)

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
        os.remove(self.dir + '/' + fname[5:])

    def RunGreys(self):
        toOpen = self.checkExistence()

        def RMSE(expected: np.ndarray, estimated: np.ndarray):
            if expected.shape != estimated.shape:
                x1 = fromarray(expected)
                x2 = fromarray(np.sum(estimated, axis=0))
                x1.show()
                x2.show()
                return 300
            diff = np.sum((expected - estimated) ** 2)
            pixSize = expected.shape[0] * expected.shape[1]
            return (diff / pixSize) ** 0.5

        for name in toOpen:
            print(name)
            fname = 'DATA/' + name
            img = cv2.imread(fname, 0)
            resolution = img.shape
            color_mode = 'Grey'
            maxp = img.max()
            minp = img.min()
            med = np.median(img)
            mean = img.mean()
            picDecomposed = self.emd.emd(img)
            imshow(picDecomposed[0])
            #x1 = fromarray(picDecomposed)
            #x1.show()
            time.sleep(6)
            numOfIMFs = picDecomposed.shape[0]
            rmse = RMSE(img, np.sum(picDecomposed, axis=0))

            self.AddToCSV(fname, color_mode, resolution, mean, med, maxp, minp, numOfIMFs, rmse)


x = Run('FirstDataFrame1.csv')
x.RunGreys()
