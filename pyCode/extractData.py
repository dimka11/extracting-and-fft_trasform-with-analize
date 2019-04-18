import csv
import pandas as pn


def csv_reader(file_obj):
    reader=pn.read_csv(file_obj, delimiter=',')
    return reader


