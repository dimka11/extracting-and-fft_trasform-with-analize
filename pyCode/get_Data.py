import csv

import pandas as pd
import numpy as np
import os

from pyCode.extractData import csv_reader
from pyCode.fourierTransform import data_transform, fft_transform

cpath = os.path.dirname(__file__)  # pycode folder (should folder where script is run)
dpath = cpath + "/../DATA/"  # path to data that above pycode folder
tpath=cpath+'/../pyCode/'


def make_array(data):
    array=[]
    for dat in data:
        mas=[dat]
        array.append(mas)
    return array

def make_one_DataArray(*transformdata):
    DATA=[]
    for datAr in transformdata:
        for data in datAr:
            DATA.append(data)
    return DATA

def csv_writer(path,data):
    np.savetxt(path, data,fmt='%s')

def make_array_Labels(label, size):
    LAB=[]
    l=label
    for i in range(0,size):
        LAB.append(l)
    return LAB


def dit_c():
    print(os.path.dirname(__file__))


if __name__ == "__main__":

    with open(dpath + "Down.csv", 'r') as f_obj:
        data = data_transform(f_obj)
        freq_Of_ShapesDown = fft_transform(data)

        rows = 40
        cols = 450
        a = np.zeros(shape=(rows, 1, cols))
        indexer = 0
        for index, i in enumerate(freq_Of_ShapesDown):
            if i.size != cols:
                b = np.zeros(cols - i.size)
                c = np.concatenate([i, b])
                a[index] = c
                continue
            a[index] = i
        np.reshape(a, a.shape + (1, ))
        print('place for debug breakpoint')

    with open(dpath + "Run(soft).csv", 'r') as f_obj2:
        data1 = data_transform(f_obj2)
        freq_Of_ShapesRun = fft_transform(data1)
    with open(dpath + "Run9.csv", 'r') as f_obj3:
        data2 = data_transform(f_obj3)
        freq_Of_ShapesRun2 = fft_transform(data2)
        runArray = make_one_DataArray(freq_Of_ShapesRun, freq_Of_ShapesRun2)
    with open(dpath + "Up.csv", 'r') as f_obj4:
        data3 = data_transform(f_obj4)
        freq_Of_ShapesUp = fft_transform(data3)
    with open(dpath + "Walk(soft).csv", 'r') as f_obj5:
        data4 = data_transform(f_obj5)
        freq_Of_Shapes4 = fft_transform(data4)
    with open(dpath + "Walking7.csv", 'r') as f_obj8:
        data7 = data_transform(f_obj8)
        freq_Of_Shapes7 = fft_transform(data7)
    with open(dpath + "Walking8.csv", 'r') as f_obj9:
        data8 = data_transform(f_obj9)
        freq_Of_Shapes8 = fft_transform(data8)
    walking = make_one_DataArray(freq_Of_Shapes4, freq_Of_Shapes7, freq_Of_Shapes8)
    with open(dpath + "Standing.csv", 'r') as f_obj10:
        data9 = data_transform(f_obj10)
        freq_Of_Stand = fft_transform(data9)
        lendat = len(freq_Of_ShapesDown) + len(runArray) + len(freq_Of_ShapesUp) + len(walking)+len(freq_Of_Stand)
        arr= make_one_DataArray(freq_Of_ShapesDown,runArray,freq_Of_ShapesUp,walking,freq_Of_Stand)
        print(np.array(a).shape, np.array(runArray).shape, np.array(freq_Of_ShapesUp).shape, np.array(walking).shape, np.array(freq_Of_Stand).shape)
        #print(a)



