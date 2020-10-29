from Develop.Interpolations import def_interpolations, imreadAndEMD, cv2
from Develop.EMD2D import EMD2D
from Develop.SRMetrices import SSIM
import pandas as pd
import os
import numpy as np
import platform


class newRun:
    def __init__(self, fname, colored=0):
        self.dir = None
        self.colored = colored
        self.fname = fname
        self.table = pd.read_csv(fname, index_col=False)
        self.platform = platform.system()
        # self.table = self.file.drop_duplicates(subset='File Name', keep=False)

        self.runner()

    def checkExistence(self):
        temp = self.table['File Name'].copy()
        temp = np.array(temp, dtype=str)
        l1 = self.getFileNames()
        if len(temp) == 0:
            return l1
        l1 = np.setdiff1d(l1, temp)
        return np.array(l1, dtype=str)

    def getFileNames(self):
        if self.platform == 'Windows':
            dirc = os.getcwd()
            dirc = dirc.replace(dirc[2], '/') + '/DATA'
            self.dir = dirc
            return np.array(os.listdir(dirc), dtype=str)
        else:
            dirc = os.getcwd() + "/DATA"
            self.dir = dirc
            return np.array(os.listdir(dirc), dtype=str)

    def AddToCSV(self, NoIMF, name, resolution, interpolation, mean_freq, var_freq, skew_freq, med_freq, kurt_freq,
                 mean_color, var_color, skew_color, med_color, kurt_color, uni_color, uni_freq, shap_color, shap_freq,
                 entrop_color, entrop_freq):
        rows = resolution[0]
        cols = resolution[1]
        to_append = pd.DataFrame({'File Name': [name],
                                  'IMF Spot': [NoIMF],
                                  'IMF Name': ['IMF ' + str(NoIMF)],
                                  'Interpolation Method': [interpolation],
                                  'Mean Frequency': [mean_freq],
                                  'Variance Frequency': [var_freq],
                                  'Median Frequency': [med_freq],
                                  'Skewness Frequency': [skew_freq],
                                  'Kurtosis Frequency': [kurt_freq],
                                  'Mean Color': [mean_color],
                                  'Variance Color': [var_color],
                                  'Median Color': [med_color],
                                  'Skewness Color': [skew_color],
                                  'Kurtosis Color': [kurt_color],
                                  'Rows': [rows],
                                  'Cols': [cols],
                                  'Entropy Color': [entrop_color],
                                  'Shapiro Color': [shap_color],
                                  'Uniformity Color': [uni_color],
                                  'Entropy Frequency': [entrop_freq],
                                  'Shapiro Frequency': [shap_freq],
                                  'Uniformity Frequency': [uni_freq]
                                  })
        self.table = self.table.append(to_append)
        self.table.to_csv(self.fname, index=False)

    @staticmethod
    def preProcess(name: str, flags=0) -> EMD2D:
        """""""""
        as for start, assuming that all images are grey
        """""
        emd, image = imreadAndEMD(name, flags)
        return emd

    @staticmethod
    def upScale_Small_IMF(imf: np.ndarray, to_shape: tuple):
        n = len(def_interpolations)
        up_scaled = np.zeros((n, to_shape[0], to_shape[1]), dtype=np.uint8)
        for i in range(n):
            up_scaled[i] = def_interpolations[i](imf.reshape((imf.shape[0], imf.shape[1], 1)), to_shape).reshape(
                to_shape)
        return up_scaled

    @staticmethod
    def getMaxSSIM(original_imf: np.ndarray, up_scaled_imfs: np.ndarray):

        def RMSE(expected: np.ndarray, estimated: np.ndarray):
            diff = ((expected - estimated) ** 2).mean() ** 0.5
            return diff

        minimum_error = 0
        spot = 0

        for i in range(up_scaled_imfs.shape[0]):
            ssim = SSIM(original_imf, up_scaled_imfs[i])
            if ssim > minimum_error:
                spot = i
                minimum_error = ssim
        interpolations = ['Gaussian', 'Bicubic', 'Bilinear', 'Lanczos5', 'Lanczos3', 'Lanczos4', 'MitchelCubic']
        return interpolations[spot]

    def doForEach(self, fname: str, flags=0):
        emd = self.preProcess(fname, flags)
        shape = (emd.shape[0], emd.shape[1])
        mean_freq = emd.MeanFrequency
        var_freq = emd.varFrequency

        for i in range(len(emd)):
            temp_imf = cv2.resize(emd(i), (int(shape[0] / 6), int(shape[1] / 6)), interpolation=cv2.INTER_LANCZOS4)
            up_scaled = self.upScale_Small_IMF(temp_imf, to_shape=shape)
            interpolation = self.getMaxSSIM(emd(i), up_scaled)

            self.AddToCSV(NoIMF=i + 1, name=fname, resolution=shape, mean_freq=mean_freq[i],
                          var_freq=var_freq[i], interpolation=interpolation, mean_color=emd.meanColor[i],
                          med_freq=emd.MedianFreq[i], skew_freq=emd.skewnessFreq[i], skew_color=emd.skewnessColor[i],
                          kurt_freq=emd.kurtosisFreq[i], kurt_color=emd.kurtosisColor[i], var_color=emd.varColor[i],
                          med_color=emd.medianColor[i])

    def runner(self):
        toOpen = self.checkExistence()
        if self.platform == 'Windows':
            toOpen = toOpen[::-1]

        for name in toOpen:
            print(name)
            self.doForEach(fname=name, flags=0)


k = newRun('interpolations.csv')
