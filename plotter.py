import cv2
import pickle
from Develop.EMD2D import EMD2D, EMD
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.signal import find_peaks
import scipy.stats as ss
import pandas as pd
import seaborn as sns

df = pd.read_csv('imputed_interpolations.csv')
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(15, 15))
sns.histplot(df, x='Mean Color', hue='IMF Spot', ax=ax[0])
sns.histplot(df, x='Mean Frequency', hue='IMF Spot', ax=ax[1])


# sns.countplot(x='Interpolation Method', data=df).set_title('Interpolation Methods Distribution')

plt.show()

"""""""""
model = pickle.load(open('Develop/random_forest_model.pkl', 'rb'))
img = cv2.imread('DATA/dog.jpg', 0)
rows = img.shape[0]
cols = img.shape[1]
img = cv2.resize(img, (int(cols/3), int(rows/3)), interpolation=cv2.INTER_LANCZOS4)
decomposed = EMD2D(img)
for i in range(len(decomposed)):
    data = [[0, decomposed.MeanFrequency[i], decomposed.varFrequency[i],
             rows/6, cols/6, decomposed.MedianFreq[i], decomposed.skewnessFreq[i], decomposed.kurtosisFreq[i],
             decomposed.meanColor[i], decomposed.varColor[i], decomposed.medianColor[i],
             decomposed.skewnessColor[i], decomposed.kurtosisColor[i]]]

    print("The best interpolation for IMF {} is ".format(i+1) + model.predict(data)[0])


x = np.linspace(-4 * np.pi, 4 * np.pi, 10000)
y = 3 * np.sin(2 * x) - np.cos(x)

# mx_points = np.where(np.abs(y - y.max()) <= 0.00001)
# mn_points = np.where(np.abs(y - y.min()) <= 0.00001)

mx_points, _ = find_peaks(y, height=0)
mn_points, _ = find_peaks(y * (-1), height=0)

upper = interpolate.Rbf(x[mx_points], y[mx_points], kind='thin_plate')
lower = interpolate.Rbf(x[mn_points], y[mn_points], kind='thin_plate')
plt.ylim(-9, 9)
plt.grid()
plt.plot(x, y, label='Signal')
plt.plot(x[mx_points], y[mx_points], 'go')
plt.plot(x[mn_points], y[mn_points], 'ro')
plt.plot(x, x * 0)
plt.plot(x, upper(x), label='Upper envelope')
plt.plot(x, lower(x), label='Lower envelope')
#plt.plot(x, (lower(x)+upper(x))/2, c='black', label='Candidate to IMF')
plt.legend()
plt.show()
"""
