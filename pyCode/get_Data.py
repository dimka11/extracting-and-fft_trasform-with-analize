import pandas as pd
import numpy as np

from pyCode.fourierTransform import data_transform, fft_transform


def transformData(array_of_freq, label):
    d={'label':label, 'frequencies':array_of_freq}
    df = pd.DataFrame(data=d)
    return df


def make_one_DataFrame(*transformdata):
    DATA=pd.concat(transformdata, ignore_index=True)
    return DATA

def csv_dict_writer(path,data):
    data.to_csv(path, index=False,doublequote=False,sep = ',' )


if __name__ =="__main__":
    with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\DATA\Down.csv", 'r') as f_obj:
        data = data_transform(f_obj)
        freq_Of_Shapes = fft_transform(data)
        trdf1 = transformData(freq_Of_Shapes, 'Down')
    with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\DATA\Run(soft).csv", 'r') as f_obj2:
        data1 = data_transform(f_obj2)
        freq_Of_Shapes1 = fft_transform(data1)
        trdf2 = transformData(freq_Of_Shapes1, 'Run')
    with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\DATA\Run9.csv", 'r') as f_obj3:
        data2 = data_transform(f_obj3)
        freq_Of_Shapes2 = fft_transform(data2)
        trdf3 = transformData(freq_Of_Shapes2, 'Run')
    with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\DATA\Up.csv", 'r') as f_obj4:
        data3 = data_transform(f_obj4)
        freq_Of_Shapes3 = fft_transform(data3)
        trdf4 = transformData(freq_Of_Shapes3, 'Up')
    with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\DATA\Walk(soft).csv", 'r') as f_obj5:
        data4 = data_transform(f_obj5)
        freq_Of_Shapes4 = fft_transform(data4)
        trdf5 = transformData(freq_Of_Shapes4, 'Walking')
    with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\DATA\Walking7.csv", 'r') as f_obj8:
        data7 = data_transform(f_obj8)
        freq_Of_Shapes7 = fft_transform(data7)
        trdf8 = transformData(freq_Of_Shapes7, 'Walking')
    with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\DATA\Walking8.csv", 'r') as f_obj9:
        data8 = data_transform(f_obj9)
        freq_Of_Shapes8 = fft_transform(data8)
        trdf9 = transformData(freq_Of_Shapes8, 'Walking')
    result = make_one_DataFrame(trdf1, trdf2, trdf3, trdf4, trdf5, trdf8, trdf9)

    csv_dict_writer('Data1.csv', result)