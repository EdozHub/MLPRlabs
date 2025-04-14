import numpy

#############################################
##
##
##     ES 7
##
##
#############################################


#Punto A
def createMatrix(m,n):
    a = numpy.arange(m, dtype=numpy.float64).reshape(m, 1)
    b = numpy.arange(n, dtype=numpy.float64).reshape(1, n)
    x = a * b
    return x

matrix = createMatrix(3,4)
#print(matrix)


#Punto B
def normalizeArray(matrix):
    arySum = matrix.sum(0)
    return matrix / arySum.reshape(1, matrix.shape[1])

matrix = normalizeArray(numpy.arange(12).reshape(3,4))
print(matrix)
