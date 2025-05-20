import load as ld
import numpy

def S1_createDictionary(tercets):
    sDict = set([]) # Vocabolario vuoto
    for tercet in tercets: #Per ogni terzina
        words = tercet.split()
        for word in words: # Per ogni parola
            sDict.add(word) # Aggiungo la parola al dizionario
    return sDict

def S1_estimateModel(hlTercets, eps):
    # Creo un vocabolario di tutte le parole presenti nel train set
    
    sDictCommon = set([]) # Vocabolario vuoto
    for cls in hlTercets: 
        ltercets = hlTercets[cls] # Ottengo le terzine di ciascuna cantica
        sDictCls = S1_createDictionary(ltercets) # Creo un vocabolario delle parole presenti nella cantica 
        sDictCommon = sDictCommon.union(sDictCls) # Aggiungo il vocabolario appena ottenuto a quello precedente
    
    h_clsLogProb = {}
    for cls in hlTercets:
        h_clsLogProb[cls] = {w: eps for w in sDictCommon}
    
    for cls in hlTercets:
        lTercets = hlTercets[cls]
        for tercet in lTercets:
            words = tercet.split()
            for w in words:
                h_clsLogProb[cls][w] += 1
    
    for cls in hlTercets:
        nWordsCls = sum(h_clsLogProb[cls].values())  # Totale parole (con smoothing)
        for w in h_clsLogProb[cls]:
            h_clsLogProb[cls][w] = numpy.log(h_clsLogProb[cls][w]) - numpy.log(nWordsCls)

    return h_clsLogProb

    


if __name__ == '__main__':
    lInf, lPur, lPar = ld.load_data()
    trInf, valInf = ld.split_data(lInf, 4)
    trPur, valPur = ld.split_data(lPur, 4)
    trPar, valPar = ld.split_data(lPar, 4)

    chapterIndexes = {"inferno": 0, "purgatorio": 1, "paradiso": 2}
    tercetsTrain = {
        "inferno": trInf,
        "purgatorio": trPur,
        "paradiso": trPar
    } 

    ### SOLUZIONE 1 ###

    S1_model = S1_estimateModel(tercetsTrain, 0.001)