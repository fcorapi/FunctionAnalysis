#GL_Quad
#Guass-Legendre Quadrature Integration Function
#Frank Corapi (fcorapi@uwaterloo.ca)
#Last Modified: 06/21/2018
#Import Directories

import numpy as np
from scipy.special import legendre
from numpy.polynomial.legendre import legroots
import numpy as np

# #Define Legendre Function
# def Legendre(x,n):
#     leg = legendre(n)
#     P_n = leg(x)
#     return P_n

def GL_Quad(integrand, lowerBound, upperBound, N, args):
    #Weight Function
    def weight(x):
        w = (2.0*(1-(x**2)))/((N*(Legendre(x, N-1)-x*Legendre(x, N)))**2)
        return w

    #Create list of roots for P_N
    rootIndex = np.zeros(N+1)
    rootIndex[N] = 1
    roots = legroots(rootIndex)

    value = 0
    for x_i in roots:
        value = weight(x_i)*integrand(((upperBound-lowerBound)/2.0)*x_i + ((upperBound+lowerBound)/2.0), *args) + value

    value = ((upperBound-lowerBound)/2.0)*value
    return value

# def test(x, N, fn):
#     val = np.cos(x)
#     return val
#
# def randomness(x):
#     return 0
#
# Nval = 10
# integralValue = GL_Quad(test, -3, 3, 20, args=(10, randomness,))
# print integralValue - 2*np.sin(3)