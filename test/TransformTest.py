import unittest
import matplotlib.pyplot as plt
import numpy as np

from pyCode.extractData import csv_reader
from pyCode.fourierTransform import work_with_coordinates, make_intervals, interval_calculation, fft_transform, \
    data_transform


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
        with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\Accelerometer.csv", 'r') as f_obj:
            reader = csv_reader(f_obj)
            reader.columns = ['timestamp', 'x', 'y', 'z']
            time = np.array(reader['timestamp'])
            df = reader['x']
            coord=work_with_coordinates(df)
            ic=interval_calculation(time, coord)
            self.assertEqual(ic.astype(int),1490)

    def test_make_array(self):
        with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\Accelerometer.csv", 'r') as f_obj:
            reader = csv_reader(f_obj)
            reader.columns = ['timestamp', 'x', 'y', 'z']
            df = reader['z']
            time = np.array(reader['timestamp'])
            coord_inter=make_intervals(time,df)
            self.assertEqual(np.array(coord_inter).size,2)

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

    def test_vector(self):
        with open(r"C:\Users\Алена\PycharmProjects\tensorflow1\Accelerometer.csv", 'r') as f_obj:
            data=data_transform(f_obj)
            df=np.array(data[0])
            plt.plot(df[0:500])
            plt.grid()
            plt.show()