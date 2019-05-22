import csv

import pandas as pd
import unittest
import matplotlib.pyplot as plt
import numpy as np

from pyCode.extractData import csv_reader
from pyCode.fourierTransform import work_with_coordinates, make_intervals, interval_calculation, fft_transform, \
    data_transform, median_filter
from pyCode.get_Data import transformData, \
    make_one_DataArray, ab


class MyTestCase(unittest.TestCase):
    def test_extracting_Data(self):
        with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\Test1.csv", 'r') as f_obj:
            reader=csv_reader(f_obj)
            reader.columns = ['timestamp', 'x', 'y', 'z']
            df = reader['x']
            self.assertEqual(df[0],6)

    def test_filter_data(self):
        with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\Accelerometer.csv", 'r') as f_obj:
            reader = csv_reader(f_obj)
            reader.columns = ['timestamp', 'x', 'y', 'z']
            df = reader['x']
            filt_coord=work_with_coordinates(df)
            first_data=df.iloc[0:500]
            first_data.plot()
            plt.plot(filt_coord[0:500])
            plt.grid()
            plt.show()

    def test_size_of_interval(self):
        with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\DATA\WALKING.csv", 'r') as f_obj:
            reader = csv_reader(f_obj)
            reader.columns = ['timestamp', 'x', 'y', 'z']
            time = np.array(reader['timestamp'])
            df = reader['x']
            coord=work_with_coordinates(df)
            ic=interval_calculation(time, coord)
            self.assertEqual(ic,1492)



    def test_make_array(self):
        with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\DATA\ACCELEROMETER.csv", 'r') as f_obj:
            reader = csv_reader(f_obj)
            reader.columns = ['timestamp', 'x', 'y', 'z']
            df = reader['z']
            time = np.array(reader['timestamp'])
            coord_inter=make_intervals(df)
            self.assertEqual(np.array(coord_inter).size,264)

    def test_frequencies(self):
        with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\Accelerometer.csv", 'r') as f_obj:
            data=data_transform(f_obj)
            freq = fft_transform(data)
            N = len(data[0])
            dat=np.array(data[0])
            FFT_data = np.fft.fft(dat)/N

            xt = np.array(FFT_data)
            K = len(xt)
            yt = freq[0]
            fig, ax = plt.subplots()
            ax.plot(np.abs(yt.T)[:K],np.abs(xt),'b')
            plt.ylim(-1, 5)
            plt.xscale('log')
            plt.grid()
            plt.show()

    def test_median_filter_data(self):
        with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\Accelerometer.csv", 'r') as f_obj:
            reader = csv_reader(f_obj)
            reader.columns = ['timestamp', 'x', 'y', 'z']
            dz=reader['x']
            df =reader['x']
            filt_coord = median_filter(df)
            first_data = dz[0:500]
            first_data.plot()
            plt.plot(filt_coord, 'r')
            plt.grid()
            plt.show()

    def test_vector(self):
        with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\Data\Run9.csv", 'r') as f_obj:
            data=data_transform(f_obj)
            df=data[0][:]
            print(df)


    def test_median_and_M_filter_data(self):
        with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\Accelerometer.csv", 'r') as f_obj:
            reader = csv_reader(f_obj)
            reader.columns = ['timestamp', 'x', 'y', 'z']
            df = reader['x']
            filt_coord = work_with_coordinates(df)
            filt_coord2=median_filter((filt_coord))
            first_data = df.iloc[0:500]
            first_data.plot()
            plt.plot(filt_coord2[0:500])
            plt.grid()
            plt.show()

    def test_freq(self):
        with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\Data\Run9.csv", 'r') as f_obj:
            data = data_transform(f_obj)
            fftDat=fft_transform(data)
            ax=fftDat[0]
            r=data[0]
            print(np.array(ax).shape, ',',np.array(r).shape)
            plt.plot(ax, np.array(r).transpose())
            plt.grid()
            plt.show()


    def test_join_data(self):
        with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\DATA\Walk(soft).csv", 'r') as f_obj:
            data=data_transform(f_obj)
            freq_Of_Shapes = fft_transform(data)
        with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\DATA\Walking1.csv", 'r') as f_obj1:
            data2=data_transform(f_obj1)
            freq_Of_Shapes2=fft_transform(data2)
        result=make_one_DataArray(freq_Of_Shapes, freq_Of_Shapes2)
        print(len(freq_Of_Shapes), len(freq_Of_Shapes2))
        print('result:', len(result))


    def test_writer(self):
             with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\DATA\WALKING.csv", 'r') as f_obj:
                 data = data_transform(f_obj)
                 freq_Of_Shapes = fft_transform(data)
                 np.savetxt('out.csv',freq_Of_Shapes)

    def test_w(self):
        with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\DATA\Walk(soft).csv", 'r') as f_obj5:
            data4 = data_transform(f_obj5)
            freq_Of_Shapes4 = fft_transform(data4)
            trdf5 = transformData(freq_Of_Shapes4, 'Walking')
        plt.plot(freq_Of_Shapes4[0], data4[0])

