import pandas as pd
import numpy as np


def transformData(array_of_freq, label):
    df = pd.DataFrame(np.array((label, array_of_freq)), columns=[label, 'frequencies'])
    return df


def make_one_DataFrame(*transformdata):
    dt=[transformdata]
    DATA=pd.concat(dt)
    return DATA