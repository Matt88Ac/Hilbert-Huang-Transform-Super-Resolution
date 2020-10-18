import cv2
from Develop.EMD2D import EMD2D

img = cv2.imread('DATA/sammy.jpg', 0)
deco = EMD2D(img)
k = deco[0]
for i in range(1, len(deco)):
    cv2.imwrite('Phase_' + str(i) + '.jpg', k)
    k += deco[i]
