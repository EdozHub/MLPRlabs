import numpy as np
import matplotlib.pyplot as plt

dic = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }

def mrow(x):
    x = np.array(x)
    return x.reshape(1, x.shape[0])

def mcol(y):
    y = np.array(y)
    return y.reshape(y.shape[0], 1)

def read_file(filename):
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

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2/3)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def compute_stats(D, L, label):
    Dclass = D[:, L==label]
    mu = np.mean(Dclass, axis=1).reshape(Dclass.shape[0], 1)
    C = (Dclass - mu) @ (Dclass - mu).T / Dclass.shape[1]
    return mu, C

def compute_likelhood():
    pass


if __name__ == "__main__":
    data, labels= read_file('lab06/iris.csv')
    (DTR, LTR), (DTE, LTE) = split_db_2to1(data, labels, seed=0)
    muSetosa, CSetosa = compute_stats(DTR, LTR, dic['Iris-setosa'])
    muVersicolor, CVersicolor = compute_stats(DTR, LTR, dic['Iris-versicolor'])
    muVirginica, CVirginica = compute_stats(DTR, LTR, dic['Iris-virginica'])
    print("muVersicolor: ", muVersicolor)
    print("CVersicolor: ", CVersicolor)