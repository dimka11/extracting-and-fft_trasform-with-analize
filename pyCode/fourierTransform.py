import numpy as np


from pyCode.extractData import csv_reader


def data_transform(file_obj, key):
    coordinates = csv_reader(file_obj)
    coordinates.columns = ['timestamp', 'x', 'y', 'z']
    time = np.array(coordinates['timestamp'])
    coord=work_with_coordinates(coordinates, key)
    array_of_coordinates_intervals=make_intervals(time,coord)
    return array_of_coordinates_intervals

def fft_transform(array_of_coordinates_intervals):
    sampling_rate = .00588
    array_of_frequencies = []
    freq = make_array_of_frequencies(array_of_coordinates_intervals, array_of_frequencies, sampling_rate)
    return freq

def make_array_of_frequencies(array_of_coordinates_intervals,array_of_frequencies,sampling_rate):
    for df in array_of_coordinates_intervals:
        FFT_data = np.fft.fft(df[:])
        freq = np.fft.fftfreq(np.array(FFT_data).shape[-1], d=sampling_rate)
        array_of_frequencies.append(freq)
    return array_of_frequencies

def make_intervals(time,coord):
    ic = interval_calculation(time, coord)
    array_of_coordinates_intervals = creating_intervals(ic, coord)
    return array_of_coordinates_intervals

def work_with_coordinates(coord, key):
    filter_value = 10
    taken_coordinates = filter_record(coord[key], filter_value)
    crd = np.array(taken_coordinates)
    return crd

def interval_calculation(tm, cd):
    tm_max = tm.max()
    points_per_second = cd.size // tm_max
    size_of_interval = points_per_second.astype(int) * 10
    return size_of_interval

def creating_intervals(int_size, coords):
    data_intervals = []
    i = 0
    size_of_interval = int_size.astype(int)
    while i < len(coords):
        chunk = coords[i:i + size_of_interval]
        data_intervals.append(chunk)
        i += size_of_interval
    return data_intervals

def data_normalisation(req, length):
    ir = np.ones(length) / length
    return np.convolve(req, ir, mode='same')

def filter_record(record, filter_value):
    filt_record = data_normalisation(record, filter_value)
    return filt_record
