from Interpolations import def_interpolations, imreadAndEMD, EMD2D
import pandas as pd
import os
import numpy as np
import platform
from itertools import combinations_with_replacement

class newRun:
    def __init__(self, fname, colored=0):
        self.dir = None
        self.colored = colored
        self.fname = fname
        self.file = pd.read_csv(fname)
        self.platform = platform.system()
        self.file = self.file.drop_duplicates(subset='File Name', keep=False)

    def checkExistence(self):
        temp = self.file['File Name'].copy()
        temp = np.array(temp, dtype=str)
        l1 = self.getFileNames()
        if len(temp) == 0:
            return l1
        l2 = ['DATA/' + y for y in l1]
        l1 = np.setdiff1d(l2, temp)
        return np.array([y[5:] for y in l1], dtype=str)

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

    @classmethod
    def getPermutations(cls, decomposed: EMD2D):
        n = len(decomposed)
        return list(combinations_with_replacement(def_interpolations, n))

