import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

def vcol(x):
    x = np.array(x)
    return x.reshape(x.shape[0], 1)

def vrow(y):
    y = np.array(y)
    return y.reshape(1, y.shape[0])

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
            attrs = vcol([float(x) for x in items[:-1]])
            label = dictionary[items[-1]]
            Dlist.append(attrs)
            Llist.append(label)
    return np.hstack(Dlist), np.array(Llist)

def compute_mu_C(D):
    mu = D.mean(1).reshape(D.shape[0], 1)
    DC = D - mu
    C = (DC @ DC.T) / (DC.shape[1])
    return mu, C

def compute_Sb_Sw(D,L):
    Sb=0
    Sw=0
    muGlobal = vcol(D.mean(1))
    for i in np.unique(L):
        DCls = D[:, L == i]
        mu = vcol(DCls.mean(1))
        Sb += (mu - muGlobal) @ (mu - muGlobal).T * DCls.shape[1]
        Sw += (DCls - mu) @ (DCls - mu).T
    return Sb / D.shape[1], Sw / D.shape[1]

def compute_lda_geig(D,L,m):
    Sb, Sw = compute_Sb_Sw(D,L)
    s, U = linalg.eigh(Sb, Sw)
    return U[:, ::-1][:, 0:m]

def compute_lda_JointDiag(D,L,m):
    Sb, Sw = compute_Sb_Sw(D,L)
    U, s, _ = np.linalg.svd(Sw)
    P = np.dot(U * vrow(1.0/(s**0.5)), U.T)

    Sb2 = np.dot(P, np.dot(Sb, P.T))
    U2, s2, _ = np.linalg.svd(Sb2)

    P2 = U2[:, 0:m]
    return np.dot(P2.T, P).T


def apply_lda(U, D):
    return U.T @ D

def plotIris(DP, L):
    plt.figure()
    plt.scatter(DP[0,L==0], DP[1,L==0], c='r', label='Iris-setosa')
    plt.scatter(DP[0,L==1], DP[1,L==1], c='g', label='Iris-versicolor')
    plt.scatter(DP[0,L==2], DP[1,L==2], c='b', label='Iris-virginica')
    plt.legend()
    plt.title('Iris LDA')
    plt.savefig('graphics/iris_lda.png')
    plt.show()

if __name__ == '__main__':
    D, L = readFile("iris.csv")
    W = compute_lda_geig(D, L, m = 2)
    DP=apply_lda(W, D)
    plotIris(DP, L)