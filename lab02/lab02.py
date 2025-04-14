import numpy as np

def load(filename):
    data = np.genfromtxt(filename, delimiter=",", usecols=range(4))
    data = data.T
    return data

array = load("iris.csv")
