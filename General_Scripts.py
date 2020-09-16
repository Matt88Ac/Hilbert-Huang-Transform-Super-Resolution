import cv2
import numpy as np
import os

Sharpen3x3 = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
Sharpen3x3 = Sharpen3x3.reshape((3, 3))


def interactiveImread(flags=None):
    path = os.getcwd()
    path = path.replace(path[2], '/')
    temp = path.split('/')
    while temp[len(temp) - 1] != 'Hilbert-Huang-Transform-Super-Resolution':
        temp.pop()
    newPath = ''

    for t in temp:
        newPath += t + '/'

    os.chdir(newPath)

    newPath += 'DATA/'
    lis = os.listdir(newPath)
    name = ''
    spot = 0

    while True:
        print("\n \n \n")
        print(str(spot + 1) + ' of ' + str(len(lis)))
        if spot >= len(lis):
            lis = os.listdir(newPath)
            spot = 0
            name = ''
            print("Error 1")
            continue

        if len(lis) == 0:
            lis = os.listdir(newPath)
            name = ''
            spot = 0
            print("Error 2")
            continue

        print("Standing on: " + lis[spot])
        print("\n ***********\nSelect current: esc\nForward: ++\nBackward: --\n ********** \n")
        inp = input("Search: ")

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
    newPath = 'DATA/' + lis[spot]
    temp = cv2.imread(newPath, flags)
    os.chdir(path)
    return temp


def imread(fname, flags):
    path = os.getcwd()
    path = path.replace(path[2], '/')
    temp = path.split('/')
    while temp[len(temp) - 1] != 'Hilbert-Huang-Transform-Super-Resolution':
        temp.pop()
    newPath = ''

    for t in temp:
        newPath += t + '/'

    os.chdir(newPath)
    temp = cv2.imread('DATA/' + fname, flags)
    os.chdir(path)
    return temp
