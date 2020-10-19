import cv2
import pickle
from Develop.EMD2D import EMD2D

model = pickle.load(open('Develop/random_forest_model.pkl', 'rb'))
img = cv2.imread('DATA/giraffes.jpg', 0)
rows = img.shape[0]
cols = img.shape[1]
img = cv2.resize(img, (int(cols/6), int(rows/6)), interpolation=cv2.INTER_LANCZOS4)
decomposed = EMD2D(img)
for i in range(len(decomposed)):
    data = [[0, decomposed.MeanFrequency[i], decomposed.varFrequency[i],
             rows/6, cols/6, decomposed.MedianFreq[i], decomposed.skewnessFreq[i], decomposed.kurtosisFreq[i],
             decomposed.meanColor[i], decomposed.varColor[i], decomposed.medianColor[i],
             decomposed.skewnessColor[i], decomposed.kurtosisColor[i]]]

    print("The best interpolation for IMF {} is ".format(i+1) + model.predict(data)[0])
