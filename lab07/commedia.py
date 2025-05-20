import load as ld
from collections import Counter
from itertools import chain

def createDictionary(data):
    words = chain.from_iterable(map(str.split, data))
    return Counter(words)

if __name__ == '__main__':
    lInf, lPur, lPar = ld.load_data()
    trInf, valInf = ld.split_data(lInf, 4)
    trPur, valPur = ld.split_data(lPur, 4)
    trPar, valPar = ld.split_data(lPar, 4)
    dicInf = createDictionary(trInf)
    dicPur = createDictionary(trPur)
    dicPar = createDictionary(trPar)