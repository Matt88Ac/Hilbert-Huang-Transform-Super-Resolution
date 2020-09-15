import cv2
import numpy as np
import os

Sharpen3x3 = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
Sharpen3x3 = Sharpen3x3.reshape((3, 3))


def imread_Interactive(flags=None):
    path = os.getcwd()
    path = path.replace(path[2], '/')
    path += '/DATA/'
    lis = os.listdir(path)
    name = ''
    spot = 0
    while True:
        print(str(spot + 1) + ' of ' + str(len(lis)))
        if spot >= len(lis):
            lis = os.listdir(path)
            spot = 0
            name = ''
            print("Error 1")
            continue

        if len(lis) == 0:
            lis = os.listdir(path)
            name = ''
            spot = 0
            print("Error 2")
            continue

        print(lis[spot])
        inp = input()

        if inp == 'esc':
            break

        elif inp == '++':
            spot += 1
            continue

        elif inp == '--':
            spot -= 1
            continue

        name += inp
        lis = list(filter(lambda x: name in x, lis))
        spot = 0
    path = 'DATA/' + lis[spot]
    return cv2.imread(path, flags)


imread_Interactive()
