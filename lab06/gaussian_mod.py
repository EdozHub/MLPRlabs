import numpy as np
import matplotlib.pyplot as plt

def mcol(x):
    x = np.array(x)
    return x.reshape(1, x.shape[0])

def mrow(y):
    y = np.array(y)
    return y.reshape(y.shape[0], 1)

def read_file(filename):
    dic = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    Dlist = []
    Llist = []
    with open(filename, 'r') as f:
        for line in f:
            items = line.strip().split(',')
            attrs = mcol([float(x) for x in items[:-1]])
            label = dic[items[-1]]
            Dlist.append(attrs)
            Llist.append(label)
    return np.hstack(Dlist), np.array(Llist)

if __name__ == "__main__":
    data, labels= read_file('iris.csv')
    print(data)
    print(labels)