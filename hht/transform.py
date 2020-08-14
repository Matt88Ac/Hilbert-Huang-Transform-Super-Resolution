import numpy as np 
import matplotlib.pyplot as plt 
from skimage.feature import peak_local_max


class HHT2D:
    def __init__(self, image):
        self.img = image
        self.IMFs = self._EMD()

    def EMD(self):
        image_max = ndimage.maximum_filter(self.img,min_distance=20)
        


        

        
