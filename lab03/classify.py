import pca
import lda
import numpy as np
import matplotlib.pyplot as plt

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

def plot_hist(D, L, title):
    plt.figure()
    plt.hist(D[0, L==1], bins=10, alpha=0.4,  label='Iris-versicolor')
    plt.hist(D[0, L==2], bins=10, alpha=0.4,  label='Iris-virginica')
    plt.savefig('graphics/classification/' + title + '.png')
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    DIris, LIris = lda.readFile('iris.csv')
    D = DIris[:, LIris != 0]
    L = LIris[LIris != 0]
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # Solution without PCA pre-processing and threshold selection. The threshold is chosen half-way between the two classes
    ULDA = lda.compute_lda_geig(DTR, LTR, m=1)

    DTR_lda = lda.apply_lda(ULDA, DTR)
    DVAL_lda  = lda.apply_lda(ULDA, DVAL)

    threshold = (DTR_lda[0, LTR==1].mean() + DTR_lda[0, LTR==2].mean()) / 2.0 # Estimated only on model training data

    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 2
    PVAL[DVAL_lda[0] < threshold] = 1
    print('Labels:     ', LVAL)
    print('Predictions:', PVAL)
    print('Number of erros:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.1f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))

    # Solution with PCA pre-processing and threshold selection. The threshold is chosen half-way between the two classes
    m=2

    UPCA = pca.compute_pca(DTR, m)

    DTR_pca = pca.apply_pca(DTR, UPCA)
    DVAL_pca  = pca.apply_pca(DVAL, UPCA)

    ULDA = lda.compute_lda_JointDiag(DTR_pca, LTR, m = 1)

    DTR_lda = lda.apply_lda(ULDA, DTR_pca)
    DVAL_lda  = lda.apply_lda(ULDA, DVAL_pca)

    threshold = (DTR_lda[0, LTR==1].mean() + DTR_lda[0, LTR==2].mean()) / 2.0

    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 2
    PVAL[DVAL_lda[0] < threshold] = 1
    print('Labels:     ', LVAL)
    print('Predictions:', PVAL)
    print('Number of erros:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.1f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))






