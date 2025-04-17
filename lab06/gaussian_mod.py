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

def logGAU_ND_singleSample(x, mu, C):
    first_term = -mu.shape[0] / 2 * np.log(2*np.pi)
    second_term = - 0.5 * np.linalg.slogdet(C)[1]
    third_term = -0.5 * (x-mu).T @ np.linalg.inv(C) @ (x-mu)
    return (first_term + second_term + third_term).ravel()

def logGAU_ND(X, mu, C):
    ll = [logGAU_ND_singleSample(X[:, i:i+1], mu, C) for i in range(X.shape[1])]
    return np.array(ll).ravel()

def compute_likelhood(X, mu, C):
    ll = logGAU_ND(X, mu, C)
    return ll

def compute_class_post_probability(score_matrix):
    post_prob = np.zeros(score_matrix.shape)
    for i in range(score_matrix.shape[0]):
        post_prob[i] = 1/3 * np.exp(score_matrix[i])
    return post_prob

def compute_score_matrix(llsetosa, llversicolor, llvirginica):
    score = []
    score.append(llsetosa)
    score.append(llversicolor)
    score.append(llvirginica)
    score_matrix = np.array(score)
    return score_matrix

if __name__ == "__main__":
    data, labels= read_file('lab06/iris.csv')
    (DTR, LTR), (DTE, LTE) = split_db_2to1(data, labels, seed=0)
    muSetosa, CSetosa = compute_stats(DTR, LTR, dic['Iris-setosa'])
    muVersicolor, CVersicolor = compute_stats(DTR, LTR, dic['Iris-versicolor'])
    muVirginica, CVirginica = compute_stats(DTR, LTR, dic['Iris-virginica'])
    llsetosa = compute_likelhood(DTR, muSetosa, CSetosa)
    llversicolor = compute_likelhood(DTR, muVersicolor, CVersicolor)
    llvirginica = compute_likelhood(DTR, muVirginica, CVirginica)
    score_matrix = compute_score_matrix(llsetosa, llversicolor, llvirginica)
    np.savetxt('lab06/score_output.csv', score_matrix.T, fmt='%.6f', delimiter=',', header='Setosa,Versicolor,Virginica', comments='')
    post_prob = compute_class_post_probability(score_matrix)