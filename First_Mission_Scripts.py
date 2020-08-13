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
        # self.table = self.table.dropna()
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

    def AddToCSV(self, fname: str, mode, resolution, mean, median, mx, mn, imfs, rmse, trace, diff0, diff1):
        to_append = pd.DataFrame({'File Name': [fname],
                                  'Color Mode': [mode],
                                  'Resolution': [resolution],
                                  'Mean Pixel Value': [mean],
                                  'Median Pixel Value': [median],
                                  'Max Pixel Value': [mx],
                                  'Min Pixel Value': [mn],
                                  'Trace': [np.log(trace)],
                                  'Difference-Axis0': [diff0],
                                  'Difference-Axis1': [diff1],
                                  'Number of IMFs': [imfs],
                                  'RMSE': rmse})
        self.table = self.table.append(to_append)
        self.table.to_csv(self.name, index=False)

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

        def Trace(pic: np.ndarray):
            mx = max(pic.shape[0], pic.shape[1])
            to_ret = np.zeros((mx, mx))
            to_ret[:pic.shape[0], :pic.shape[1]] = pic.copy()
            return np.trace(to_ret)

        def Diffs(pic: np.ndarray):
            return abs(np.diff(pic.astype(int), axis=0)).max(), abs(np.diff(pic.astype(int), axis=1)).max()

        for name in toOpen:
            print(name)
            fname = 'DATA/' + name  # 'road_image.jpg'
            img = cv2.imread(fname, 0)
            resolution = img.shape
            color_mode = 'Grey'
            maxp = img.max()
            minp = img.min()
            med = np.median(img)
            mean = img.mean()
            picDecomposed = self.emd.emd(img)
            # X, Y = self.emd.find_extrema(img)
            # print('X = ', X)
            # print('Y = ', Y)
            # print(self.emd.find_extrema(img))
            x1 = fromarray(picDecomposed[0])
            x1.show()
            return
            tr = Trace(img)
            dif0, dif1 = Diffs(img)
            numOfIMFs = picDecomposed.shape[0]
            rmse = RMSE(img, np.sum(picDecomposed, axis=0))

            self.AddToCSV(fname=fname, mode=color_mode, resolution=resolution,
                          mean=mean, median=med, mx=maxp, mn=minp, imfs=numOfIMFs, rmse=rmse, trace=tr,
                          diff0=dif0, diff1=dif1)


x = Run('FirstDataFrame1.csv')
x.RunGreys()
