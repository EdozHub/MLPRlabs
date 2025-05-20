import load as ld

if __name__ == '__main__':
    lInf, lPur, lPar = ld.load_data()
    trInf, valInf = ld.split_data(lInf, 4)
    trPur, valPur = ld.split_data(lPur, 4)
    trPar, valPar = ld.split_data(lPar, 4)