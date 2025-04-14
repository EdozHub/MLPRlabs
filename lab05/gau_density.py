import numpy as np
import matplotlib.pyplot as plt

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def logpdf_GAU_ND_singleSample(x, mu, C):
    first_term = -mu.shape[0] / 2 * np.log(2*np.pi)
    second_term = - 0.5 * np.linalg.slogdet(C)[1]
    third_term = -0.5 * (x-mu).T @ np.linalg.inv(C) @ (x-mu)
    return (first_term + second_term + third_term).ravel()

def logpdf_GAU_ND(X, mu, C):
    ll = [logpdf_GAU_ND_singleSample(X[:, i:i+1], mu, C) for i in range(X.shape[1])]
    return np.array(ll).ravel()

def loglikehood(X, mu, C):
    ll = logpdf_GAU_ND(X, mu, C)
    return ll.sum()

if __name__ == "__main__":
    plt.figure()
    XPlot = np.linspace(-8, 12, 1000)
    m = np.ones((1,1)) * 1.0
    C = np.ones((1,1)) * 2.0
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), m, C)))
    plt.savefig("graphics/gaussian.png")
    plt.show()

    pdfSol = np.load('Solution/llGAU.npy')
    pdfGau = logpdf_GAU_ND(vrow(XPlot), m, C)   
    print(np.abs(pdfSol - pdfGau).max())

    XND = np.load('Solution/XND.npy')
    mu = np.load('Solution/muND.npy')
    C = np.load('Solution/CND.npy')
    pdfSol = np.load('Solution/llND.npy')
    pdfGau = logpdf_GAU_ND(XND, mu, C)
    print(np.abs(pdfSol - pdfGau).max())

    #ML - ESTIMATION - XND
    mu_ML, C_ML = compute_mu_C(XND)
    print("ML mu: ", mu_ML)
    print("ML C: ", C_ML)
    ll = loglikehood(XND, mu_ML, C_ML)
    print("ML loglikehood: ", ll)

    #ML -ESTIMATION - X1D
    X1D = np.load('Solution/X1D.npy')
    mu_ML, C_ML = compute_mu_C(X1D)
    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = np.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), mu_ML, C_ML)))
    plt.savefig("graphics/MLGaussianX1D.png")
    plt.show()

    ll = loglikehood(X1D, mu_ML, C_ML)
    print(ll)