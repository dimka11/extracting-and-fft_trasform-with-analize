import numpy as np
from scipy import signal

from pyCode.extractData import csv_reader


def data_transform(file_obj):
    coordinates = csv_reader(file_obj)
    coordinates.columns = ['timestamp', 'x', 'y', 'z']
    coordV = coordinates['magnitude'] = np.sqrt(coordinates['x']**2 + coordinates['y']**2 + coordinates['y']**2)
    return make_intervals(coordV)

def fft_transform(vector_of_accelerations):
    freq = make_array_of_frequencies(vector_of_accelerations)
    return freq

def make_array_of_frequencies (vector_of_accelerations):
    sampling_rate = .00588
    array_of_frequencies = []
    for df in vector_of_accelerations:
            FFT_data = np.fft.fft(df[:])
            freq = np.fft.fftfreq(np.array(FFT_data).shape[-1], d=sampling_rate)
            array_of_frequencies.append(np.abs(freq))
    return array_of_frequencies

def make_intervals(coord):##Разбиение данных на интервалы
    ic = 500
    array_of_coordinates_intervals = creating_intervals(ic, coord)
    return array_of_coordinates_intervals

def work_with_coordinates(coord):###Фильтрация данных и преобзазование их в массив
    filter_value = 5
    taken_coordinates = filter_record(coord, filter_value)
    crd = np.array(taken_coordinates)
    return crd

def interval_calculation(tm, cd):
    tm_max = (tm.max()-tm.min())/1000
    points_per_second = cd.size / tm_max
    size_of_interval = points_per_second * 10
    return size_of_interval.astype(int)

def creating_intervals(int_size, coords):
    data_intervals = []
    i = 0
    size_of_interval = int_size
    while i < len(coords):
        chunk = coords[i:i + size_of_interval]
        data_intervals.append(chunk)
        i += size_of_interval
    return data_intervals

def data_normalisation(req, length):
    ir = np.ones(length) / length
    return np.convolve(req, ir, mode='same')

def filter_record(record, filter_value):
    filt_rec = data_normalisation(record, filter_value)
    filt_record=median_filter(filt_rec)
    return filt_record

def create_vector_of_acceleration(array_of_coordinates_intervals_X, array_of_coordinates_intervals_Y, array_of_coordinates_intervals_Z):
    arrvectors=[]
    vector=[]
    for arr in range(len(array_of_coordinates_intervals_X)):###он получает список массивов
        x = np.array(array_of_coordinates_intervals_X[arr])
        y = np.array(array_of_coordinates_intervals_Y[arr])
        z = np.array(array_of_coordinates_intervals_Z[arr])
        for element_index in range(len(x)):
            v = np.math.sqrt((np.math.pow(x[element_index], 2) + np.math.pow(y[element_index], 2) + np.math.pow(z[element_index], 2)))
            vector.append(v)
        arrvectors.append(vector)
    return arrvectors

def median_filter(data):
    dat=np.array(data)
    f_data=signal.medfilt(dat)
    return f_data
