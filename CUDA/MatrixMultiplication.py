import numpy as np
from timeit import default_timer as timer
from numba import vectorize

@vectorize(["float64(float64,float64)"], target = 'cuda')
def MultiplyVectors(VectorA, VectorB):
    return VectorA * VectorB


def VectorMultiply(a, b, c):

    for i in range(a.size):
        c[i] = a[i] * b[i]


def main():
    Size = 64000000

    MatrixA = np.ones(Size)
    MatrixB = np.ones(Size)
    MatrixC = np.ones(Size)

    start = timer()
    VectorMultiply(MatrixA, MatrixB, MatrixC)
    duration = timer() - start

    print("MatrixC[:6] = " + str(MatrixC[:6]))
    print("MatrixC[-6:] = " + str(MatrixC[-6:]))
    print("The computation using CPU was completed in:", "{0:.2f}".format(duration),"seconds.\n")

    MatrixD = np.ones(Size)
    MatrixE = np.ones(Size)
    MatrixF = np.ones(Size)

    start = timer()
    MatrixF = MultiplyVectors(MatrixA, MatrixB)
    duration = timer() - start

    print("MatrixF[:6] = " + str(MatrixF[:6]))
    print("MatrixF[-6:] = " + str(MatrixF[-6:]))
    print("The computation using CUDA was completed in:", "{0:.2f}".format(duration),"seconds.\n")


main()
