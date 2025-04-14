import numpy as np
import matplotlib.pyplot as plt

def mcol(x):
    x = np.array(x)
    return x.reshape(x.shape[0], 1)

def readFile(filename):
    dictionary = {
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
            label = dictionary[items[-1]]
            Dlist.append(attrs)
            Llist.append(label)
    return np.hstack(Dlist), np.array(Llist)

def compute_pca(D,m):
    mu = D.mean(1).reshape(D.shape[0],1)
    DC = D - mu
    C = (DC @ DC.T) / (DC.shape[1])
    U, s, Vh = np.linalg.svd(C)
    P = U[:,0:m]
    return P

def apply_pca(D, P):
    return P.T @ D

def plot_pca(P, L):
    plt.figure()
    plt.scatter(P[0, L == 0], P[1, L == 0], color='red', label='Iris-setosa')
    plt.scatter(P[0, L == 1], P[1, L == 1], color='green', label='Iris-versicolor')
    plt.scatter(P[0, L == 2], P[1, L == 2], color='blue', label='Iris-virginica')
    plt.title('PCA of Iris Dataset')
    plt.savefig('graphics/iris_pca.png')
    plt.show()

if __name__ == "__main__":
    D,L = readFile('iris.csv')
    P = compute_pca(D, 2)
    Psol = apply_pca(D, P)
    plot_pca(Psol, L)
    

