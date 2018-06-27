#GL_Quad_2D
#Guass-Legendre Quadrature Integration Function in 2 Dimensions
#Frank Corapi (fcorapi@uwaterloo.ca)
#Last Modified: 06/25/2018
#Import Directories

import numpy as np
from scipy.special import legendre
from numpy.polynomial.legendre import legroots


#Define Legendre Function
def Legendre(x,n):
    leg = legendre(n)
    P_n = leg(x)
    return P_n

def GL_Quad_2D(integrand, lowZ, upZ, lowPhi, upPhi, N, args):
    #Weight Function
    def weight(arg):
        w = (2.0*(1-(arg**2)))/((N*(Legendre(arg, N-1)-arg*Legendre(arg, N)))**2)
        return w

    #Create list of roots for P_N
    rootIndex = np.zeros(N+1)
    rootIndex[N] = 1
    roots = legroots(rootIndex)

    #Equally spaced points for trapzeoidal integration method for phi
    Npoints = 100
    deltaPhi = (upPhi-lowPhi)/Npoints
    phiVals = np.linspace(lowPhi, upPhi, Npoints+1)

    value = 0
    for z_i in roots:
        phiValue = 0
        for phi in phiVals:
            phiValue = (deltaPhi) * integrand(((upZ-lowZ)/2.0)*z_i + ((upZ+lowZ)/2.0), phi, *args) + phiValue
        value = weight(z_i)*phiValue + value

    value = ((upZ-lowZ)/2.0)*value
    return value

def test(z, phi, N, fn):
    val = z**2*np.sin(phi)**2
    return val

def randomness(x):
    return 0

Nval = 10
integralValue = GL_Quad_2D(test, -1, 1, 0, 2*np.pi, 30, args=(10, randomness,))
print integralValue - 2*np.pi/3
