#SphericalHarmonicsDE
#This script takes a given function and represents it as a series of Spherical Harmonics,
#and will eventually work to solve DEs.
#Frank Corapi (fcorapi@uwaterloo.ca)
#Last Modified: 08/02/2018

#Import Directories
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre, sph_harm, roots_legendre
from scipy.integrate import quad
from numpy.polynomial.legendre import legroots
from pylab import imshow
from mpl_toolkits import mplot3d
import time

#***********************************FUNCTION DEFINITIONS*************************************

#Define Legendre Function
def Legendre(x,n):
    leg = legendre(n)
    P_n = leg(x)
    return P_n

#*****************************SCALAR SPHERICAL HARMONICS************************************
#Function that we want to represent as a Legendre Series
#TO BE MODIFIED BY USER DEPENDING ON WHICH FUNCTION IS WANTED
def DesiredFunction(theta,phi):
    val = sph_harm(0, 3, phi, theta)
    return np.real(val)

#Derivative of the desired function
#NEEDS TO BE FIXED
# def DerivFunction(x):
#     val = 20.0*(-20.0)*np.sin(20*x)
#     return val

#Analytic result for the phi function
#TO BE MODIFIED BY USER DEPENDING ON WHICH FUNCTION IS WANTED
def phi(theta, phi):
    val = 0
    for l in range(0, 25):
        for m in np.arange(-l, l, 2):
            val = sph_harm(m, l, phi, theta) + val
    return np.real(val)

#Mass Density Function Example (rho)
#TO BE MODIFIED BY USER DEPENDING ON WHICH FUNCTION IS WANTED
def rho(theta, phi):
    val = 0
    for l in range(1, 25):
        for m in np.arange(-l, l, 2):
            val = -l*(l+1)*sph_harm(m, l, phi, theta) + val
    return np.real(val)

#Function to be integrated to determine SH coefficients
def integrand(z, phi, n, m, fn):
    value = np.conj(sph_harm(m, n, phi, np.arccos(z)))*fn(np.arccos(z), phi)
    return value

def findCoeff(Nval, fn, intTerms):
    coeffList = []
    # Integrate to determine Legendre series coefficients
    for n in range(0, Nval):
        for m in range(-n, n+1):
            integralValue = GL_Quad_2D(integrand, -1.0, 1.0, 0, 2*np.pi, intTerms, 1, args=(n, m, fn,))
            cval = integralValue
            coeffList.append(cval)
    return coeffList

#SHSeries
def SHSeries(z, phi, N, coeff):
    series = 0
    ntracker = 0

    if N > len(coeff):
        print "Error"
        return 0
    else:
        for loopn in range(0, N):
            mtracker = 0
            for loopm in range(-loopn, loopn+1):
                series = series + coeff[mtracker + ntracker]*sph_harm(loopm, loopn, phi, np.arccos(z))
                mtracker = mtracker + 1
            ntracker = (2*loopn + 1) + ntracker
    return series

#SHSeries
def SHSeriesAng(theta, phi, N, coeff):
    series = 0
    ntracker = 0

    if N > len(coeff):
        print "Error"
        return 0
    else:
        for loopn in range(0, N):
            mtracker = 0
            for loopm in range(-loopn, loopn+1):
                series = series + coeff[mtracker + ntracker]*sph_harm(loopm, loopn, phi, theta)
                mtracker = mtracker + 1
            ntracker = (2*loopn + 1) + ntracker
    return series

#Calculate the norm of a particular Legendre polynomial
def LegNorm(n):
    norm = 2.0/(2.0*n+1)
    return norm

#Establish the differentiation matrix for the Legendre polynomials (derived analytically).
def DerivMatrix(Nsize):
    D = np.zeros((Nsize,Nsize))
    i = 0
    while i < (Nsize):
        j = 0
        while (2*j+i+1) < (Nsize):
            D[i,(2*j+i+1)] = 1.0/LegNorm(i)
            j = j + 1
        i = i + 1
    return 2*D

def LMatrix(Nsize):
    L = np.matmul(DerivMatrix(Nsize), DerivMatrix(Nsize))

    #Apply first boundary condition (x=1)
    L[Nsize-2,:] = 1

    #Apply second boundary condition (x=-1)
    for loop in range(0,Nsize):
        if loop%2 == 0:
            L[Nsize-1, loop] = 1
        else:
            L[Nsize-1, loop] = -1
    return L

def LaplaceMatrix(Nsize):
    lap = np.zeros((Nsize, Nsize))
    n = 0
    m = 0
    for loopn in range(0, Nsize):
        if m >= 2*n+1:
            m = 0
            n = n + 1
        if loopn == 0:
            lap[loopn, loopn] = 1
        else:
            lap[loopn, loopn] = -n*(n+1)
        m = m + 1
    return lap

#print LaplaceMatrix(7)

def calcDeriv(order, coeffs):
    matSize = len(coeffs)
    primes = coeffs
    if order <= 0:
        return primes
    else:
        for loop in range(1, order+1):
            primes = np.matmul(DerivMatrix(matSize), primes)
        return primes

#Function to integrate over to find error in the Legendre Series
def L2ErrorFunction(z, phi, N, coeff, fn):
    errVal = abs(SHSeries(z, phi, N, coeff) - fn(np.arccos(z), phi))**2
    return errVal

#Loops over every N value up to a maximum, and calculates the L2 error.
def calcErrorList(coeff, Nval, fn, intTerms):
    errList = []
    for maxN in range(1, Nval + 1):
        err = GL_Quad_2D(L2ErrorFunction, -1.0, 1.0, 0, 2*np.pi, intTerms, 0, args=(maxN, coeff, fn,))
        errList.append(np.sqrt(err))
        print "Error for N = ", maxN, " completed."
    return errList

def GL_Quad(integrand, lowerBound, upperBound, N, args):
    #Weight Function
    def weight(x):
        w = (2.0*(1-(x**2)))/((N*(Legendre(x, N-1)-x*Legendre(x, N)))**2)
        return w

    #Create list of roots for P_N
    # rootIndex = np.zeros(N+1)
    # rootIndex[N] = 1
    # roots = legroots(rootIndex)
    roots = roots_legendre(N)[0]

    value = 0
    for x_i in roots:
        value = weight(x_i)*integrand(((upperBound-lowerBound)/2.0)*x_i + ((upperBound+lowerBound)/2.0), *args) + value

    value = ((upperBound-lowerBound)/2.0)*value
    return value

def GL_Quad_2D(integrand, lowZ, upZ, lowPhi, upPhi, N, round, args):

    if round == 1:
        def conformalFactor(z,phi):
            return 1
    elif round == 0:
        def conformalFactor(z,phi):
            return psi4(z,phi)

    #Weight Function
    def weight(arg):
        w = (2.0*(1-(arg**2)))/((N*(Legendre(arg, N-1)-arg*Legendre(arg, N)))**2)
        return w

    #Create list of roots for P_N
    # rootIndex = np.zeros(N+1)
    # rootIndex[N] = 1
    # roots = legroots(rootIndex)
    roots = roots_legendre(N)[0]

    #Equally spaced points for trapzeoidal integration method for phi
    Npoints = 30
    deltaPhi = (upPhi-lowPhi)/Npoints
    phiVals = np.linspace(lowPhi, upPhi-deltaPhi, Npoints)

    # Zpoints = 1000
    # deltaZ = (upZ-lowZ)/Zpoints
    # Zvals = np.linspace(lowZ, upZ, Zpoints+1)
    # ztracker = 0

    value = 0
    for z_i in roots:
        phiValue = 0
        for phi in phiVals:
                phiValue = deltaPhi * conformalFactor(z_i,phi) * integrand(((upZ - lowZ) / 2.0) * z_i + ((upZ + lowZ) / 2.0), phi, *args) + phiValue
        value = weight(z_i)*phiValue + value

    value = ((upZ-lowZ)/2.0)*value
    return value

#*********************************************VECTOR SPHERICAL HARMONICS*************************************
#Define conformal factor
def psi4(z,phi):
    factor = 1# - z**2
    return factor

#*****OLD METHOD - Finite Differences*******
# #Partial Phi derivative of a spherical harmonic
# def phiDerivSH(M, N, phi, theta):
#     eps = 1e-5
#     deriv = (sph_harm(M, N, phi + 0.5*eps, theta) - sph_harm(M, N, phi - 0.5*eps, theta))/eps
#     return deriv

#Partial Phi derivative of a spherical harmonic
def phiDerivSH(M, N, phi, theta):
    deriv = (1j*M)*sph_harm(M, N, phi, theta)
    return deriv

#Second Partial Phi derivative of a spherical harmonic
def phiSecondDerivSH(M, N, phi, theta):
    deriv = (-1)*(M**2)*sph_harm(M, N, phi, theta)
    return deriv

#*****OLD METHOD - Finite Differences*******
# #Partial Theta derivative of a spherical harmonic
# def thetaDerivSH(M, N, phi, theta):
#     eps = 1e-5
#     deriv = (sph_harm(M, N, phi, theta + 0.5*eps) - sph_harm(M, N, phi, theta - 0.5*eps))/eps
#     return deriv

#Partial Theta derivative of a spherical harmonic
def thetaDerivSH(M, N, phi, theta):
    if M == N:
        deriv = M*(np.cos(theta)/np.sin(theta))*sph_harm(M, N, phi, theta)
    else:
        deriv = M * (np.cos(theta) / np.sin(theta)) * sph_harm(M, N, phi, theta) + np.sqrt(
            (N - M) * (N + M + 1)) * np.exp(-1j * phi) * sph_harm(M+1, N, phi, theta)
    return deriv

#Partial Theta derivative of a spherical harmonic
def thetaSecondDerivSH(M, N, phi, theta):
    if M == N:
        deriv = M*(np.cos(theta)/np.sin(theta))*sph_harm(M, N, phi, theta)
    elif M == N-1:
        deriv = M*(M*((np.cos(theta) / np.sin(theta))**2) - ((1/np.sin(theta))**2)) * sph_harm(M, N, phi, theta) \
                + np.sqrt((N-M)*(N+M+1))*(2*M + 1)*np.exp(-1j*phi)*(np.cos(theta) / np.sin(theta)) * sph_harm(M+1, N, phi, theta)
    else:
        deriv = M * (M * ((np.cos(theta) / np.sin(theta)) ** 2) - ((1 / np.sin(theta)) ** 2)) * sph_harm(M, N, phi, theta) \
                + np.sqrt((N-M)*(N+M+1))*(2*M + 1)*np.exp(-1j*phi)*(np.cos(theta) / np.sin(theta)) * sph_harm(M+1, N, phi, theta) \
                + np.sqrt((N-M)*(N-M-1)*(M+N+2)*(M+N+1))*np.exp(-2j*phi)*sph_harm(M+2, N, phi, theta)
    return deriv

#Partial Theta derivative of a spherical harmonic
def thetaPhiDerivSH(M, N, phi, theta):
    if M == N:
        deriv = M*(np.cos(theta)/np.sin(theta))*sph_harm(M, N, phi, theta)
    else:
        deriv = M * (np.cos(theta) / np.sin(theta)) * sph_harm(M, N, phi, theta) + np.sqrt(
            (N - M) * (N + M + 1)) * np.exp(-1j * phi) * sph_harm(M+1, N, phi, theta)
    return (1j*M)*deriv

def VecDesiredFunction(theta,phi,kind):
    A = 2 * np.sqrt((np.pi) / 3)
    if kind == 'polar':
        val = [np.cos(theta), 0*theta]
    elif kind == 'axial':
        A = 2 * np.sqrt((2 * np.pi) / 3)
        val = np.add([0.5j * A * phiDerivSH(1, 1, phi, theta) / np.sin(theta),
                      -0.5j * A * np.sin(theta) * thetaDerivSH(1, 1, phi, theta)],
                     [0.5j * A * phiDerivSH(-1, 1, phi, theta) / np.sin(theta),
                      -0.5j * A * np.sin(theta) * thetaDerivSH(-1, 1, phi, theta)])
    return val

#Representing Phi1 co-vector field from Korzynski paper using vector SH
def Phi1(theta, phi, kind):
    if kind == 'axial':
        A = 2*np.sqrt((2*np.pi)/3)*psi4(np.cos(theta),phi)
        val = np.subtract([-0.5 * A * phiDerivSH(1, 1, phi, theta)/np.sin(theta), 0.5 * A * np.sin(theta) * thetaDerivSH(1, 1, phi, theta)],
                          [-0.5 * A * phiDerivSH(-1, 1, phi, theta)/np.sin(theta), 0.5 * A * np.sin(theta) * thetaDerivSH(-1, 1, phi, theta)])
    elif kind == 'polar':
        val = [0,0]
        print "Error, Phi1 is not a polar vector!"
    return val

#Representing Phi2 co-vector field from Korzynski paper using vector SH
def Phi2(theta, phi, kind):
    if kind == 'axial':
        A = 2 * np.sqrt((2 * np.pi) / 3)*psi4(np.cos(theta),phi)
        val = np.add([0.5j * A * phiDerivSH(1, 1, phi, theta)/np.sin(theta), -0.5j * A * np.sin(theta) * thetaDerivSH(1, 1, phi, theta)],
                     [0.5j * A * phiDerivSH(-1, 1, phi, theta)/np.sin(theta), -0.5j * A * np.sin(theta) * thetaDerivSH(-1, 1, phi, theta)])
    elif kind == 'polar':
        val = [0,0]
        print "Error, Phi2 is not a polar vector!"
    return val

#Representing Phi3 co-vector field from Korzynski paper using vector SH
def Phi3(theta, phi, kind):
    if kind == 'axial':
        A = 2 * np.sqrt((np.pi) / 3)*psi4(np.cos(theta),phi)
        val = [A * phiDerivSH(0, 1, phi, theta)/np.sin(theta), -A * np.sin(theta) * thetaDerivSH(0, 1, phi, theta)]
    elif kind == 'polar':
        val = [0,0]
        print "Error, Phi3 is not a polar vector!"
    return val

#Representing Xi1 co-vector field from Korzynski paper using vector SH
def Xi1(theta, phi, kind):
    if kind == 'polar':
        A = 2 * np.sqrt((2 * np.pi) / 3)*psi4(np.cos(theta),phi)
        val = np.subtract([0.5 * A * thetaDerivSH(1, 1, phi, theta), 0.5 * A * phiDerivSH(1, 1, phi, theta)],
                          [0.5 * A * thetaDerivSH(-1, 1, phi, theta), 0.5 * A * phiDerivSH(-1, 1, phi, theta)])
    elif kind == 'axial':
        val = [0,0]
        print "Error, Xi1 is not an axial vector!"
    return val

#Representing Xi2 co-vector field from Korzynski paper using vector SH
def Xi2(theta, phi, kind):
    if kind == 'polar':
        A = 2 * np.sqrt((2 * np.pi) / 3)*psi4(np.cos(theta),phi)
        val = np.add([-0.5j * A * thetaDerivSH(1, 1, phi, theta), -0.5j * A * phiDerivSH(1, 1, phi, theta)],
                     [-0.5j * A * thetaDerivSH(-1, 1, phi, theta), -0.5j * A * phiDerivSH(-1, 1, phi, theta)])
    elif kind == 'axial':
        val = [0,0]
        print "Error, Xi2 is not an axial vector!"
    return val

#Representing Xi3 co-vector field from Korzynski paper using vector SH
def Xi3(theta, phi, kind):
    if kind == 'polar':
        A = 2 * np.sqrt((np.pi) / 3)*psi4(np.cos(theta),phi)
        val = [-A * thetaDerivSH(0, 1, phi, theta), -A * phiDerivSH(0, 1, phi, theta)]
    elif kind == 'axial':
        val = [0,0]
        print "Error, Xi3 is not an axial vector!"
    return val

def vectorSH(M, N, phi, theta, kind):
    if kind == 'polar':
        vec = [thetaDerivSH(M, N, phi, theta), phiDerivSH(M, N, phi, theta)]
    elif kind == 'axial':
        vec = [phiDerivSH(M, N, phi, theta)/np.sin(theta), -np.sin(theta)*thetaDerivSH(M, N, phi, theta)]
    return vec

def vecIntegrand(z, phi, n, m ,fn, kind):
    #Define the spherical metric
    q_inv = np.zeros((2, 2))
    q_inv[0, 0] = 1
    q_inv[1, 1] = 1/(np.sin(np.arccos(z))**2)

    #Inner product involving the spherical metric
    value = 0
    for loop1 in range(0,2):
        for loop2 in range(0,2):
            value = value + q_inv[loop1, loop2]*np.conj(vectorSH(m, n, phi, np.arccos(z), kind))[loop1]*fn(np.arccos(z), phi, kind)[loop2]
    #value = np.dot(np.conj(vectorSH(m, n, phi, np.arccos(z), kind)), fn(np.arccos(z), phi, kind))
    return value

def findVecCoeff(Nval, fn, intTerms, kind):
    coeffList = []
    # Integrate to determine Legendre series coefficients
    for n in range(0, Nval):
        for m in range(-n, n+1):
            integralValue = GL_Quad_2D(vecIntegrand, -1.0, 1.0, 0, 2*np.pi, intTerms, 1, args=(n, m, fn, kind,))
            if n == 0:
                cval = integralValue
            else:
                cval = integralValue / (n * (n + 1))
            coeffList.append(cval)
    return coeffList

#VecSHSeries
def VecSHSeries(z, phi, N, coeff, kind):
    series = np.zeros(np.shape(vectorSH(0,0,phi,np.arccos(z), kind)))
    ntracker = 0

    if N > len(coeff):
        print "Error"
        return 0
    else:
        for loopn in range(0, N):
            mtracker = 0
            for loopm in range(-loopn, loopn+1):
                series = series + np.multiply((coeff[mtracker + ntracker]), vectorSH(loopm, loopn, phi, np.arccos(z), kind))
                mtracker = mtracker + 1
            ntracker = (2*loopn + 1) + ntracker
    return series

#Function to integrate over to find error in the SH Series
def L2VecErrorFunction(z, phi, N, coeff, fn, kind):
    # Define the spherical metric
    q_inv = np.zeros((2, 2))
    q_inv[0, 0] = 1
    q_inv[1, 1] = 1 / (np.sin(np.arccos(z))**2)

    diff = VecSHSeries(z, phi, N, coeff, kind) - fn(np.arccos(z), phi, kind)
    errVal = 0

    for loop1 in range(0,2):
        for loop2 in range(0,2):
            errVal = errVal + (1/psi4(z,phi))*q_inv[loop1, loop2]*np.conj(diff)[loop1]*diff[loop2]
    #errVal = abs(np.dot(np.conj(VecSHSeries(z, phi, N, coeff, kind) - fn(np.arccos(z), phi, kind)), VecSHSeries(z, phi, N, coeff, kind) - fn(np.arccos(z), phi, kind)))
    return abs(errVal)

#Loops over every N value up to a maximum, and calculates the L2 error.
def calcVecErrorList(coeff, Nval, fn, intTerms, kind):
    errList = []
    for maxN in range(1, Nval + 1):
        err = GL_Quad_2D(L2VecErrorFunction, -1.0, 1.0, 0, 2*np.pi, intTerms, 0, args=(maxN, coeff, fn, kind,))
        errList.append(np.sqrt(err))
        print "Error for N = ", maxN, " completed."
    return errList

#*********************************************TENSOR SPHERICAL HARMONICS*************************************

def tensorSH(M, N, phi, theta, kind):
    tens = np.zeros([2, 2], dtype=np.complex_)
    X_lm = 2*(thetaPhiDerivSH(M,N,phi,theta) - (np.cos(theta)/np.sin(theta))*phiDerivSH(M,N,phi,theta))
    W_lm = thetaSecondDerivSH(M,N,phi,theta) - (np.cos(theta)/np.sin(theta))*thetaDerivSH(M,N,phi,theta) \
           - ((1/np.sin(theta))**2)*phiSecondDerivSH(M,N,phi,theta)
    if kind == 'metric':
        tens[0,0] = sph_harm(M, N, phi, theta)
        tens[0,1] = 0+0j
        tens[1,0] = 0+0j
        tens[1,1] = (np.sin(theta)**2)*sph_harm(M, N, phi, theta)
    elif kind == 'polar':
        tens[0,0] = 0.5*W_lm
        tens[0,1] = 0.5*X_lm
        tens[1,0] = 0.5*X_lm
        tens[1,1] = -0.5*(np.sin(theta)**2)*W_lm
    elif kind == 'levi':
        tens[0,0] = 0+0j
        tens[0,1] = np.sin(theta)*sph_harm(M, N, phi, theta)
        tens[1,0] = -np.sin(theta)*sph_harm(M, N, phi, theta)
        tens[1,1] = 0+0j
    elif kind == 'axial':
        tens[0,0] = (0.5/np.sin(theta))*X_lm
        tens[0,1] = -0.5*np.sin(theta)*W_lm
        tens[1,0] = -0.5*np.sin(theta)*W_lm
        tens[1,1] = -0.5*np.sin(theta)*X_lm
    return tens

def TensDesiredFunction(theta,phi,kind):
    A = 0.25 * np.sqrt(5/(np.pi))
    B = 1/np.sqrt(4*np.pi)
    if kind == 'polar':
        val = np.zeros([2,2],dtype=np.complex_)
        val[0,0] = 3*A*(np.sin(theta)**2)
        val[0,1] = 0+0j
        val[1,0] = 0+0j
        val[1,1] = -3*A*(np.sin(theta)**4)
    elif kind == 'axial':
        val = np.zeros([2, 2], dtype=np.complex_)
        val[0, 0] = 0 + 0j
        val[0, 1] = -3 * A * (np.sin(theta) ** 3)
        val[1, 0] = -3 * A * (np.sin(theta) ** 3)
        val[1, 1] = 0 + 0j
    elif kind == 'metric':
        val = np.zeros([2, 2], dtype=np.complex_)
        val[0, 0] = sph_harm(2, 3, phi, theta) + sph_harm(-1, 1, phi, theta) + sph_harm(-1, 2, phi, theta)
        val[0, 1] = 0+0j
        val[1, 0] = 0+0j
        val[1, 1] = (sph_harm(2, 3, phi, theta) + sph_harm(-1, 1, phi, theta) + sph_harm(-1, 2, phi, theta))*np.sin(theta)**2
    elif kind == 'levi':
        val = np.zeros([2, 2], dtype=np.complex_)
        val[0, 0] = 0 + 0j
        val[0, 1] = (sph_harm(2, 3, phi, theta) + sph_harm(-1, 1, phi, theta) + sph_harm(-1, 2, phi, theta))*np.sin(theta)
        val[1, 0] = -(sph_harm(2, 3, phi, theta) + sph_harm(-1, 1, phi, theta) + sph_harm(-1, 2, phi, theta))*np.sin(theta)
        val[1, 1] = 0 + 0j
    return val

def tensIntegrand(z, phi, n, m ,fn, kind):
    #Define the spherical metric
    q_inv = np.zeros((2, 2))
    q_inv[0, 0] = 1
    q_inv[1, 1] = 1/(np.sin(np.arccos(z))**2)

    #Inner product involving the spherical metric
    value = 0
    for loop1 in range(0,2):
        for loop2 in range(0,2):
            for loop3 in range(0,2):
                for loop4 in range(0,2):
                    value = value + q_inv[loop1, loop3]*q_inv[loop2, loop4]*np.conj(tensorSH(m, n, phi, np.arccos(z), kind))[loop1,loop2]*fn(np.arccos(z), phi, kind)[loop3,loop4]
    #value = np.dot(np.conj(vectorSH(m, n, phi, np.arccos(z), kind)), fn(np.arccos(z), phi, kind))
    return value

def findTensCoeff(Nval, fn, intTerms, kind):
    coeffList = []
    # Integrate to determine Legendre series coefficients
    for n in range(0, Nval):
        for m in range(-n, n+1):
            integralValue = GL_Quad_2D(tensIntegrand, -1.0, 1.0, 0, 2*np.pi, intTerms, 1, args=(n, m, fn, kind,))
            if kind == 'metric':
                cval = integralValue/2.0
            elif kind == 'levi':
                cval = integralValue/2.0
            else:
                if n == 0:
                    cval = integralValue
                elif n ==1:
                    cval = integralValue
                else:
                    cval = integralValue / (0.5*(n-1)*n*(n+1)*(n+2))
            coeffList.append(cval)
    return coeffList

#TensSHSeries
def TensSHSeries(z, phi, N, coeff, kind):
    series = np.zeros(np.shape(tensorSH(0,0,phi,np.arccos(z), kind)),dtype=np.complex_)
    ntracker = 0

    if N > len(coeff):
        print "Error"
        return 0
    else:
        for loopn in range(0, N):
            mtracker = 0
            for loopm in range(-loopn, loopn+1):
                series = series + np.multiply((coeff[mtracker + ntracker]), tensorSH(loopm, loopn, phi, np.arccos(z), kind))
                mtracker = mtracker + 1
            ntracker = (2*loopn + 1) + ntracker
    return series

#Function to integrate over to find error in the SH Series
def L2TensErrorFunction(z, phi, N, coeff, fn, kind):
    # Define the spherical metric
    q_inv = np.zeros((2, 2))
    q_inv[0, 0] = 1
    q_inv[1, 1] = 1 / (np.sin(np.arccos(z))**2)

    diff = TensSHSeries(z, phi, N, coeff, kind) - fn(np.arccos(z), phi, kind)
    errVal = 0

    for loop1 in range(0,2):
        for loop2 in range(0,2):
            for loop3 in range(0,2):
                for loop4 in range(0,2):
                    errVal = errVal + (1/psi4(z,phi))*(1/psi4(z,phi))*q_inv[loop1, loop3]*q_inv[loop2, loop4]*np.conj(diff)[loop1,loop2]*diff[loop3,loop4]
    #errVal = abs(np.dot(np.conj(VecSHSeries(z, phi, N, coeff, kind) - fn(np.arccos(z), phi, kind)), VecSHSeries(z, phi, N, coeff, kind) - fn(np.arccos(z), phi, kind)))
    return abs(errVal)

#Loops over every N value up to a maximum, and calculates the L2 error.
def calcTensErrorList(coeff, Nval, fn, intTerms, kind):
    errList = []
    for maxN in range(1, Nval + 1):
        err = GL_Quad_2D(L2TensErrorFunction, -1.0, 1.0, 0, 2*np.pi, intTerms, 0, args=(maxN, coeff, fn, kind,))
        errList.append(np.sqrt(err))
        print "Error for N = ", maxN, " completed."
    return errList

#******************Functions needed to calculate angular momentum integrals************************

#Function returning one
def oneFunction(z, phi):
    return 1

#Define the one-form omega used for angular momentum calculations
def omega(theta, phi, N, wcoeff, kind):
    val = VecSHSeries(np.cos(theta), phi, N, wcoeff, kind)
    return val

def JKIntegrand(z, phi, w, vecFn, N, wcoeff, wKind, fnKind):
    # Define the spherical metric
    q_inv = np.zeros((2, 2))
    q_inv[0, 0] = 1
    q_inv[1, 1] = 1 / (np.sin(np.arccos(z)) ** 2)

    # Inner product involving the spherical metric
    value = 0
    for loop1 in range(0, 2):
        for loop2 in range(0, 2):
            value = value + (1/psi4(z,phi)) * q_inv[loop1, loop2] * w(np.arccos(z), phi, N, wcoeff, wKind)[loop1] * vecFn(np.arccos(z), phi, fnKind)[loop2]
    # value = np.dot(np.conj(vectorSH(m, n, phi, np.arccos(z), kind)), fn(np.arccos(z), phi, kind))
    return value

def calculateJK(w, vecFn, N, intTerms, wcoeff, wKind, fnKind):

    integralValue = GL_Quad_2D(JKIntegrand, -1.0, 1.0, 0, 2 * np.pi, intTerms, 0, args=(w, vecFn, N, wcoeff, wKind, fnKind,))

    return integralValue
#*******************************END OF FUNCTIONS*************************************

Nval = 5 #Number of coefficients
intN = 2*Nval #Number of terms in Gauss-Legendre integration
thetaVals = np.linspace(0, np.pi, 100) + 1e-5#Theta-Values
phiVals = np.linspace(0, 2*np.pi, 100) + 1e-5 #Phi-Values
theta_mesh, phi_mesh = np.meshgrid(thetaVals, phiVals) #Make a mesh grid
X_mesh = np.sin(theta_mesh)*np.cos(phi_mesh)
Y_mesh = np.sin(theta_mesh)*np.sin(phi_mesh)
Z_mesh = np.cos(theta_mesh)
coeffNum = np.linspace(0,Nval-1,Nval) #List of N-values
G = 1 #Graviational constant
c = 1 #Speed of light
# w,v = np.linalg.eig(LaplaceMatrix(17))
# print w
# print v

#***********Representing Desired Function*****************
# t = time.time()
# #print sph_harm(0,0,0,0)
# print "Finding coefficients..."
# C_n = findCoeff(Nval, DesiredFunction, intN) #Coefficients of Desired Function
# print "Coefficients Found!"
# # Cprime_n = calcDeriv(2, C_n) #Coefficients of the derivative of the function
#
# checkCoeff = []
# for check in range(len(C_n)):
#     if abs(C_n[check]) > 1e-6:
#         checkCoeff.append(1)
#     else:
#         checkCoeff.append(0)
#
# #List L2 error for each N-value.
# print "Calculating Error List..."
# errorList = calcErrorList(C_n, Nval, DesiredFunction, intN)
# # derivErrorList = calcErrorList(Cprime_n, Nval, DerivFunction, intN)
# error = errorList[len(errorList)-1]
# # derivError = derivErrorList[len(derivErrorList)-1]
# print "Errors Calculated!"
#
# print "Determining Series..."
# seriesResult = SHSeries(np.cos(theta_mesh), phi_mesh, Nval, C_n)
# print "Series determined, plotting results..."
# # derivSeriesResult = SHSeries(thetaVals, phiVals, Nval, Cprime_n)
#
# #Print Results
# print "Spherical Harmonics Series Coefficients:", np.real(C_n)
# print "Checking Values of Coeffecients:", checkCoeff
# print "Error:", error
# # print "Derivative Legendre Series Coefficients", Cprime_n
# # print "Derivative Error:", derivError
#
# elapsedTime = time.time() - t
# print "Elapsed Time (s):", elapsedTime
#
# #Plotting Results
# ax = plt.axes(projection='3d')
# ax.plot_surface(theta_mesh, phi_mesh, np.real(seriesResult), cmap = 'viridis', edgecolor='none')
# ax.plot_surface(theta_mesh, phi_mesh, DesiredFunction(theta_mesh, phi_mesh), edgecolor='none')
# ax.set_title('Spherical Harmonics Series')
# ax.set_xlabel('Theta-Values')
# ax.set_ylabel('Phi-Values')
# plt.show(ax)
#
# plt.figure()
# plt.contourf(theta_mesh, phi_mesh, np.real(seriesResult)-DesiredFunction(theta_mesh, phi_mesh), 30, cmap='hot')
# plt.colorbar()
# plt.title('Error Plot')
# plt.xlabel('Theta-Values')
# plt.ylabel('Phi-Values')
# plt.show()
#
# # plt.figure()
# # plt.contourf(theta_mesh, phi_mesh, np.imag(seriesResult), 30, cmap='hot')
# # plt.colorbar()
# # plt.title('Imaginary Values Plot')
# # plt.xlabel('Theta-Values')
# # plt.ylabel('Phi-Values')
# #
# # plt.figure()
# # plt.contourf(theta_mesh, phi_mesh, np.real(seriesResult), 30, cmap='hot')
# # plt.colorbar()
# # plt.title('Real Values Plot')
# # plt.xlabel('Theta-Values')
# # plt.ylabel('Phi-Values')
#
# #Scatter plot the L2 error versus N
# plt.figure()
# plt.scatter(coeffNum, np.log10(errorList))
# #plt.yscale('log')
# plt.xlabel('N-Value')
# plt.ylabel('Log_10 of L2 Error')
# plt.grid()
# plt.title('L2 Error for Different N-Values')
# plt.show()

#*****************************************************************************************

#***********Representing Desired Vector Function*****************
# t = time.time()
# vecKind = 'polar' #The Kind has to be 'polar' or 'axial'
#
# print "Finding coefficients..."
# C_n = findVecCoeff(Nval, VecDesiredFunction, intN, vecKind) #Coefficients of Desired Function
# print "Coefficients Found!"
# # Cprime_n = calcDeriv(2, C_n) #Coefficients of the derivative of the function
#
# checkCoeff = []
# for check in range(len(C_n)):
#     if abs(C_n[check]) > 1e-6:
#         checkCoeff.append(1)
#     else:
#         checkCoeff.append(0)
#
# #List L2 error for each N-value.
# print "Calculating Error List..."
# errorList = calcVecErrorList(C_n, Nval, VecDesiredFunction, intN, vecKind)
# # derivErrorList = calcErrorList(Cprime_n, Nval, DerivFunction, intN)
# error = errorList[len(errorList)-1]
# # derivError = derivErrorList[len(derivErrorList)-1]
# print "Errors Calculated!"
#
# print "Determining Series..."
# seriesResult = VecSHSeries(np.cos(theta_mesh), phi_mesh, Nval, C_n, vecKind)
# print np.imag(seriesResult)
# print "Series determined, plotting results..."
# # derivSeriesResult = SHSeries(thetaVals, phiVals, Nval, Cprime_n)
#
# #Print Results
# print "Spherical Harmonics Series Coefficients:", C_n
# print "Checking Values of Coeffecients:", checkCoeff
# print "Error:", error
# # print "Derivative Legendre Series Coefficients", Cprime_n
# # print "Derivative Error:", derivError
#
#
# #Calculating the J and K values from the Korzynski paper.
# J1 = (-1/(8*np.pi*G))*calculateJK(omega, Phi1, Nval, intN, C_n, vecKind, 'axial')
# J2 = (-1/(8*np.pi*G))*calculateJK(omega, Phi2, Nval, intN, C_n, vecKind, 'axial')
# J3 = (-1/(8*np.pi*G))*calculateJK(omega, Phi3, Nval, intN, C_n, vecKind, 'axial')
# K1 = (-1/(8*np.pi*G))*calculateJK(omega, Xi1, Nval, intN, C_n, vecKind, 'polar')
# K2 = (-1/(8*np.pi*G))*calculateJK(omega, Xi2, Nval, intN, C_n, vecKind, 'polar')
# K3 = (-1/(8*np.pi*G))*calculateJK(omega, Xi3, Nval, intN, C_n, vecKind, 'polar')
#
# #Calculate the values of the invariants A and B from the  Korzynski paper
# invariantA = (J1*J1 + J2*J2 + J3*J3) - (K1*K1 + K2*K2 + K3*K3)
# invariantB = K1*J1 + K2*J2 + K3*J3
#
# #Final value for the Angular Momentum
# J = np.sqrt((invariantA + np.sqrt(invariantA**2 + 4*(invariantB**2)))/2)
#
# #Calculate properties of the sphere
# area = GL_Quad_2D(oneFunction, -1, 1, 0, 2*np.pi, intN, 0, args=())
# arealRadius = np.sqrt(area/(4*np.pi))
# irrMass = arealRadius/2
# mass = np.sqrt(irrMass**2 + J**2/(4*(irrMass**2)))
# spin = J/(mass**2)
#
# #Printing the results of the calculated J and K values.
# print "The omega one-form is equal to the co-vector phi_3."
# print "J_1 = ", J1
# print "J_2 = ", J2
# print "J_3 = ", J3
# print "K_1 = ", K1
# print "K_2 = ", K2
# print "K_3 = ", K3
#
# print "A = ", invariantA
# print "B = ", invariantB
# print "J = ", J
#
# print "Properties of the Surface:"
# print "Area: ", area
# print "Areal Radius: ", arealRadius
# print "Irreducible Mass: ", irrMass
# print "Mass: ", mass
# print "Spin (a): ", spin
#
# elapsedTime = time.time() - t
# print "Elapsed Time (s):", elapsedTime
#
# #Scatter plot the L2 error versus N
# plt.figure()
# plt.scatter(coeffNum, np.log10(errorList))
# #plt.yscale('log')
# plt.xlabel('N-Value')
# plt.ylabel('Log_10 of L2 Error')
# plt.grid()
# plt.title('L2 Error for Different N-Values $(\\phi_3)$')
# plt.show()
#
# #Plotting the vector fields using quiver
# plt.figure()
# plt.quiver(phi_mesh[::4,::4], theta_mesh[::4,::4], seriesResult[1,::4,::4], -seriesResult[0,::4,::4])
# plt.xlabel('$\\phi$-Values')
# plt.ylabel('$\\theta$-Values')
# plt.gca().invert_yaxis()
# plt.title('Vector Spherical Harmonics Series Plot $(\\phi_3)$')
# plt.figure()
# plt.quiver(phi_mesh[::4,::4], theta_mesh[::4,::4], VecDesiredFunction(theta_mesh, phi_mesh, vecKind)[1][::4,::4],
#            -VecDesiredFunction(theta_mesh, phi_mesh, vecKind)[0][::4,::4], color = 'r')
# plt.xlabel('$\\phi$-Values')
# plt.ylabel('$\\theta$-Values')
# plt.gca().invert_yaxis()
# plt.title('Vector Desired Function Plot $(\\phi_3)$')
# plt.show()

#*************************************************************************************************************

#***********Representing Desired Tensor Field*****************
# t = time.time()
# tensKind = 'levi' #The Kind has to be 'polar', 'axial', 'metric, or 'levi'
#
# print "Finding coefficients..."
# C_n = findTensCoeff(Nval, TensDesiredFunction, intN, tensKind) #Coefficients of Desired Function
# print "Coefficients Found!"
# # Cprime_n = calcDeriv(2, C_n) #Coefficients of the derivative of the function
#
# checkCoeff = []
# for check in range(len(C_n)):
#     if abs(C_n[check]) > 1e-6:
#         checkCoeff.append(1)
#     else:
#         checkCoeff.append(0)
#
# #List L2 error for each N-value.
# print "Calculating Error List..."
# errorList = calcTensErrorList(C_n, Nval, TensDesiredFunction, intN, tensKind)
# # derivErrorList = calcErrorList(Cprime_n, Nval, DerivFunction, intN)
# error = errorList[len(errorList)-1]
# # derivError = derivErrorList[len(derivErrorList)-1]
# print "Errors Calculated!"
#
# #print "Determining Series..."
# # seriesResult = TensSHSeries(np.cos(theta_mesh), phi_mesh, Nval, C_n, tensKind)
# # print np.imag(seriesResult)
# #print "Series determined, plotting results..."
# # derivSeriesResult = SHSeries(thetaVals, phiVals, Nval, Cprime_n)
#
# #Print Results
# print "Spherical Harmonics Series Coefficients:", C_n
# print "Checking Values of Coeffecients:", checkCoeff
# print "Error:", error
#
# elapsedTime = time.time() - t
# print "DONE!"
# print "Elapsed Time (s):", elapsedTime
#
# # #Scatter plot the L2 error versus N
# print "Plotting error..."
# plt.figure()
# plt.scatter(coeffNum, np.log10(errorList))
# #plt.yscale('log')
# plt.xlabel('N-Value')
# plt.ylabel('Log_10 of L2 Error')
# plt.grid()
# plt.title('L2 Error for Different N-Values $(\\phi_3)$')
# plt.show()

#***************Solving Poissons Equation***************
# t = time.time()
# print "Finding coefficients..."
# rho_n = findCoeff(Nval, rho, intN)
# rho_0 = findCoeff(1, phi, intN)[0] #Specified by the user.
# rho_n_solver = rho_n[:]
# rho_n_solver[0] = rho_0
# print "Coefficients Found!"
#
# print "Solving Laplace equation..."
# phi_n = np.linalg.solve(LaplaceMatrix(len(rho_n_solver)), rho_n_solver)
# print "Solved, calculating error list..."
#
# checkCoeff = []
# for check in range(len(phi_n)):
#     if abs(phi_n[check]) > 1e-6:
#         checkCoeff.append(1)
#     else:
#         checkCoeff.append(0)
#
# #List L2 error for each N-value.
# phiErrorList = calcErrorList(phi_n, Nval, phi, intN)
# phiError = phiErrorList[len(phiErrorList)-1]
# print "Errors Calculated!"
#
# #Series Solution for Phi
# print "Determining Series..."
# phiSeries = SHSeries(np.cos(theta_mesh), phi_mesh, Nval, phi_n)
# print "Series determined, plotting results..."
#
#
# print "Phi Coefficients:", np.real(phi_n)
# print "Checking Values of Coeffecients:", checkCoeff
# print "Phi Error:", phiError
#
# elapsedTime = time.time() - t
# print "Elapsed Time (s):", elapsedTime
#
# #Plotting Results
# ax = plt.axes(projection='3d')
# ax.plot_surface(theta_mesh, phi_mesh, np.real(phiSeries), cmap = 'viridis', edgecolor='none')
# ax.plot_surface(theta_mesh, phi_mesh, phi(theta_mesh, phi_mesh), edgecolor='none')
# ax.set_title('Spherical Harmonics Series for Phi-Function')
# ax.set_xlabel('Theta-Values')
# ax.set_ylabel('Phi-Values')
# plt.show(ax)
#
# plt.figure()
# plt.contourf(theta_mesh, phi_mesh, np.real(phiSeries)-phi(theta_mesh, phi_mesh), 30, cmap='hot')
# plt.colorbar()
# plt.title('Error Plot')
# plt.xlabel('Theta-Values')
# plt.ylabel('Phi-Values')
# plt.show()
#
# # plt.figure()
# # plt.contourf(theta_mesh, phi_mesh, np.imag(phiSeries), 30, cmap='hot')
# # plt.colorbar()
# # plt.title('Imaginary Values Plot')
# # plt.xlabel('Theta-Values')
# # plt.ylabel('Phi-Values')
# #
# # plt.figure()
# # plt.contourf(theta_mesh, phi_mesh, np.real(phiSeries), 30, cmap='hot')
# # plt.colorbar()
# # plt.title('Real Values Plot')
# # plt.xlabel('Theta-Values')
# # plt.ylabel('Phi-Values')
#
# #Scatter plot the phi L2 error versus N
# plt.figure()
# plt.scatter(coeffNum, np.log10(phiErrorList))
# #plt.yscale('log')
# plt.xlabel('N-Value')
# plt.ylabel('Log_10 of Phi L2 Error')
# plt.grid()
# plt.title('Phi L2 Error for Different N-Values')
# plt.show()

#******************************************************************************


#**************Representing Vectors Component-Wise Using Scalar SH Series*********************************
#*********Functions**********************************
def projOperator(theta, phi):
    proj = np.zeros((3,3))

    proj[0,0] = 1 - (np.sin(theta)**2)*(np.cos(phi)**2)
    proj[0,1] = -(np.sin(theta)**2)*np.cos(phi)*np.sin(phi)
    proj[0,2] = -np.sin(theta)*np.cos(theta)*np.cos(phi)
    proj[1,0] = -(np.sin(theta)**2)*np.cos(phi)*np.sin(phi)
    proj[1,1] = 1 - (np.sin(theta)**2)*(np.sin(phi)**2)
    proj[1,2] = -np.sin(theta)*np.cos(theta)*np.sin(phi)
    proj[2,0] = -np.sin(theta)*np.cos(theta)*np.cos(phi)
    proj[2,1] = -np.sin(theta)*np.cos(theta)*np.sin(phi)
    proj[2,2] = 1 - np.cos(theta)**2

    return proj

#Test Vector Field!
def vecField(theta, phi):
    vec = [1,0,0]
    return vec

#Test Metric Field
def metricField(theta, phi):
    metric = np.zeros([3,3])
    metric[0,0] = 1.0
    metric[1,1] = 2.0
    metric[2,2] = 3.0
    return metric

#Test Metric Field
def identity(theta, phi):
    identity = np.zeros([3,3])
    identity[0,0] = 1.0
    identity[1,1] = 1.0
    identity[2,2] = 1.0
    return identity

def invMetricField(theta,phi):
    inverseMetric = np.zeros([3, 3])
    inverseMetric[0, 0] = 1.0
    inverseMetric[1, 1] = 1.0/2.0
    inverseMetric[2, 2] = 1.0/3.0
    return inverseMetric

def vecProj(theta, phi, vector, comp):
    newVec = np.zeros(len(vector(theta,phi)))

    for loop1 in range(0,len(vector(theta,phi))):
        for loop2 in range(0,len(vector(theta,phi))):
            newVec[loop1] = projOperator(theta, phi)[loop1,loop2]*vector(theta,phi)[loop2] + newVec[loop1]
    if comp == 0:
        return newVec[0]
    elif comp == 1:
        return newVec[1]
    elif comp == 2:
        return newVec[2]

def metricProj(theta, phi, metric, comp):
    newMetric = np.zeros(np.shape(metric(theta,phi)))

    for loop1 in range(0, np.shape(metric(theta,phi))[0]):
        for loop2 in range(0, np.shape(metric(theta, phi))[1]):
            for loop3 in range(0, np.shape(metric(theta, phi))[0]):
                for loop4 in range(0, np.shape(metric(theta, phi))[1]):
                    newMetric[loop1,loop2] = projOperator(theta, phi)[loop1,loop3]*projOperator(theta, phi)[loop2,loop4]*metric(theta, phi)[loop3,loop4] + newMetric[loop1,loop2]
    if comp == 00:
        return newMetric[0,0]
    elif comp == 01:
        return newMetric[0,1]
    elif comp == 02:
        return newMetric[0,2]
    elif comp == 10:
        return newMetric[1,0]
    elif comp == 11:
        return newMetric[1,1]
    elif comp == 12:
        return newMetric[1,2]
    elif comp == 20:
        return newMetric[2,0]
    elif comp == 21:
        return newMetric[2,1]
    elif comp == 22:
        return newMetric[2,2]

def findCartCoeff(Nval, fn, intTerms, vec, comp):
    coeffList = []
    # Integrate to determine Legendre series coefficients
    for n in range(0, Nval):
        for m in range(-n, n+1):
            integralValue = GL_Quad_2D(cartIntegrand, -1.0, 1.0, 0, 2*np.pi, intTerms, 1, args=(n, m, fn, vec, comp,))
            cval = integralValue
            coeffList.append(cval)
    return coeffList

def cartIntegrand(z, phi, n, m, fn, vec, comp):
    value = np.conj(sph_harm(m, n, phi, np.arccos(z)))*fn(np.arccos(z), phi, vec, comp)
    return value

#Function to integrate over to find error in the Legendre Series
def L2CartErrorFunction(z, phi, N, coeff, fn, vec, comp):
    errVal = abs(SHSeries(z, phi, N, coeff) - fn(np.arccos(z), phi, vec, comp))**2
    return errVal

#Loops over every N value up to a maximum, and calculates the L2 error.
def calcCartErrorList(coeff, Nval, fn, intTerms, vec, cart):
    errList = []
    for maxN in range(1, Nval + 1):
        err = GL_Quad_2D(L2CartErrorFunction, -1.0, 1.0, 0, 2*np.pi, intTerms, 0, args=(maxN, coeff, fn, vec, cart,))
        errList.append(np.sqrt(err))
        print "Error for N = ", maxN, " completed."
    return errList


#************End of Functions**********************


t = time.time()
print 'Finding coefficients...'

C_nXX = findCartCoeff(Nval, metricProj, intN, metricField, 00)
print 'XX Coefficients Found!'
C_nXY = findCartCoeff(Nval, metricProj, intN, metricField, 01)
print 'XY Coefficients Found!'
C_nXZ = findCartCoeff(Nval, metricProj, intN, metricField, 02)
print 'XZ Coefficients Found!'
C_nYX = findCartCoeff(Nval, metricProj, intN, metricField, 10)
print 'YX Coefficients Found!'
C_nYY = findCartCoeff(Nval, metricProj, intN, metricField, 11)
print 'YY Coefficients Found!'
C_nYZ = findCartCoeff(Nval, metricProj, intN, metricField, 12)
print 'YZ Coefficients Found!'
C_nZX = findCartCoeff(Nval, metricProj, intN, metricField, 20)
print 'ZX Coefficients Found!'
C_nZY = findCartCoeff(Nval, metricProj, intN, metricField, 21)
print 'ZY Coefficients Found!'
C_nZZ = findCartCoeff(Nval, metricProj, intN, metricField, 22)
print 'ZZ Coefficients Found!'

print 'Finding Inverse Coefficients...'
InvC_nXX = findCartCoeff(Nval, metricProj, intN, invMetricField, 00)
print 'XX Coefficients Found!'
InvC_nXY = findCartCoeff(Nval, metricProj, intN, invMetricField, 01)
print 'XY Coefficients Found!'
InvC_nXZ = findCartCoeff(Nval, metricProj, intN, invMetricField, 02)
print 'XZ Coefficients Found!'
InvC_nYX = findCartCoeff(Nval, metricProj, intN, invMetricField, 10)
print 'YX Coefficients Found!'
InvC_nYY = findCartCoeff(Nval, metricProj, intN, invMetricField, 11)
print 'YY Coefficients Found!'
InvC_nYZ = findCartCoeff(Nval, metricProj, intN, invMetricField, 12)
print 'YZ Coefficients Found!'
InvC_nZX = findCartCoeff(Nval, metricProj, intN, invMetricField, 20)
print 'ZX Coefficients Found!'
InvC_nZY = findCartCoeff(Nval, metricProj, intN, invMetricField, 21)
print 'ZY Coefficients Found!'
InvC_nZZ = findCartCoeff(Nval, metricProj, intN, invMetricField, 22)
print 'ZZ Coefficients Found!'

checkCoeffxx = []
for check in range(len(C_nXX)):
    if abs(C_nXX[check]) > 1e-6:
        checkCoeffxx.append(1)
    else:
        checkCoeffxx.append(0)
checkCoeffxy = []
for check in range(len(C_nXY)):
    if abs(C_nXY[check]) > 1e-6:
        checkCoeffxy.append(1)
    else:
        checkCoeffxy.append(0)
checkCoeffxz = []
for check in range(len(C_nXZ)):
    if abs(C_nXZ[check]) > 1e-6:
        checkCoeffxz.append(1)
    else:
        checkCoeffxz.append(0)
checkCoeffyx = []
for check in range(len(C_nYX)):
    if abs(C_nYX[check]) > 1e-6:
        checkCoeffyx.append(1)
    else:
        checkCoeffyx.append(0)
checkCoeffyy = []
for check in range(len(C_nYY)):
    if abs(C_nYY[check]) > 1e-6:
        checkCoeffyy.append(1)
    else:
        checkCoeffyy.append(0)
checkCoeffyz = []
for check in range(len(C_nYZ)):
    if abs(C_nYZ[check]) > 1e-6:
        checkCoeffyz.append(1)
    else:
        checkCoeffyz.append(0)
checkCoeffzx = []
for check in range(len(C_nZX)):
    if abs(C_nZX[check]) > 1e-6:
        checkCoeffzx.append(1)
    else:
        checkCoeffzx.append(0)
checkCoeffzy = []
for check in range(len(C_nZY)):
    if abs(C_nZY[check]) > 1e-6:
        checkCoeffzy.append(1)
    else:
        checkCoeffzy.append(0)
checkCoeffzz = []
for check in range(len(C_nZZ)):
    if abs(C_nZZ[check]) > 1e-6:
        checkCoeffzz.append(1)
    else:
        checkCoeffzz.append(0)

#Checking Inverse Coefficients
invCheckCoeffxx = []
for check in range(len(InvC_nXX)):
    if abs(InvC_nXX[check]) > 1e-6:
        invCheckCoeffxx.append(1)
    else:
        invCheckCoeffxx.append(0)
invCheckCoeffxy = []
for check in range(len(InvC_nXY)):
    if abs(InvC_nXY[check]) > 1e-6:
        invCheckCoeffxy.append(1)
    else:
        invCheckCoeffxy.append(0)
invCheckCoeffxz = []
for check in range(len(InvC_nXZ)):
    if abs(InvC_nXZ[check]) > 1e-6:
        invCheckCoeffxz.append(1)
    else:
        invCheckCoeffxz.append(0)
invCheckCoeffyx = []
for check in range(len(InvC_nYX)):
    if abs(InvC_nYX[check]) > 1e-6:
        invCheckCoeffyx.append(1)
    else:
        invCheckCoeffyx.append(0)
invCheckCoeffyy = []
for check in range(len(InvC_nYY)):
    if abs(InvC_nYY[check]) > 1e-6:
        invCheckCoeffyy.append(1)
    else:
        invCheckCoeffyy.append(0)
invCheckCoeffyz = []
for check in range(len(InvC_nYZ)):
    if abs(InvC_nYZ[check]) > 1e-6:
        invCheckCoeffyz.append(1)
    else:
        invCheckCoeffyz.append(0)
invCheckCoeffzx = []
for check in range(len(InvC_nZX)):
    if abs(InvC_nZX[check]) > 1e-6:
        invCheckCoeffzx.append(1)
    else:
        invCheckCoeffzx.append(0)
invCheckCoeffzy = []
for check in range(len(InvC_nZY)):
    if abs(InvC_nZY[check]) > 1e-6:
        invCheckCoeffzy.append(1)
    else:
        invCheckCoeffzy.append(0)
invCheckCoeffzz = []
for check in range(len(InvC_nZZ)):
    if abs(InvC_nZZ[check]) > 1e-6:
        invCheckCoeffzz.append(1)
    else:
        invCheckCoeffzz.append(0)

#List L2 error for each N-value.
print "Calculating Error List..."
errorListxx = calcCartErrorList(C_nXX, Nval, metricProj, intN, metricField, 00)
print "Error for XX-Component Done!"
errorListxy = calcCartErrorList(C_nXY, Nval, metricProj, intN, metricField, 01)
print "Error for XY-Component Done!"
errorListxz = calcCartErrorList(C_nXZ, Nval, metricProj, intN, metricField, 02)
print "Error for XZ-Component Done!"
errorListyx = calcCartErrorList(C_nYX, Nval, metricProj, intN, metricField, 10)
print "Error for YX-Component Done!"
errorListyy = calcCartErrorList(C_nYY, Nval, metricProj, intN, metricField, 11)
print "Error for YY-Component Done!"
errorListyz = calcCartErrorList(C_nYZ, Nval, metricProj, intN, metricField, 12)
print "Error for YZ-Component Done!"
errorListzx = calcCartErrorList(C_nZX, Nval, metricProj, intN, metricField, 20)
print "Error for ZX-Component Done!"
errorListzy = calcCartErrorList(C_nZY, Nval, metricProj, intN, metricField, 21)
print "Error for ZY-Component Done!"
errorListzz = calcCartErrorList(C_nZZ, Nval, metricProj, intN, metricField, 22)
print "Error for ZZ-Component Done!"

#List L2 error for each N-value.
print "Calculating Inverse Error List..."
invErrorListxx = calcCartErrorList(InvC_nXX, Nval, metricProj, intN, invMetricField, 00)
print "Error for XX-Component Done!"
invErrorListxy = calcCartErrorList(InvC_nXY, Nval, metricProj, intN, invMetricField, 01)
print "Error for XY-Component Done!"
invErrorListxz = calcCartErrorList(InvC_nXZ, Nval, metricProj, intN, invMetricField, 02)
print "Error for XZ-Component Done!"
invErrorListyx = calcCartErrorList(InvC_nYX, Nval, metricProj, intN, invMetricField, 10)
print "Error for YX-Component Done!"
invErrorListyy = calcCartErrorList(InvC_nYY, Nval, metricProj, intN, invMetricField, 11)
print "Error for YY-Component Done!"
invErrorListyz = calcCartErrorList(InvC_nYZ, Nval, metricProj, intN, invMetricField, 12)
print "Error for YZ-Component Done!"
invErrorListzx = calcCartErrorList(InvC_nZX, Nval, metricProj, intN, invMetricField, 20)
print "Error for ZX-Component Done!"
invErrorListzy = calcCartErrorList(InvC_nZY, Nval, metricProj, intN, invMetricField, 21)
print "Error for ZY-Component Done!"
invErrorListzz = calcCartErrorList(InvC_nZZ, Nval, metricProj, intN, invMetricField, 22)
print "Error for ZZ-Component Done!"

errorxx = errorListxx[len(errorListxx)-1]
errorxy = errorListxy[len(errorListxy)-1]
errorxz = errorListxz[len(errorListxz)-1]
erroryx = errorListyx[len(errorListyx)-1]
erroryy = errorListyy[len(errorListyy)-1]
erroryz = errorListyz[len(errorListyz)-1]
errorzx = errorListzx[len(errorListzx)-1]
errorzy = errorListzy[len(errorListzy)-1]
errorzz = errorListzz[len(errorListzz)-1]

invErrorxx = invErrorListxx[len(invErrorListxx)-1]
invErrorxy = invErrorListxy[len(invErrorListxy)-1]
invErrorxz = invErrorListxz[len(invErrorListxz)-1]
invErroryx = invErrorListyx[len(invErrorListyx)-1]
invErroryy = invErrorListyy[len(invErrorListyy)-1]
invErroryz = invErrorListyz[len(invErrorListyz)-1]
invErrorzx = invErrorListzx[len(invErrorListzx)-1]
invErrorzy = invErrorListzy[len(invErrorListzy)-1]
invErrorzz = invErrorListzz[len(invErrorListzz)-1]
print "Errors Calculated!"

print "Determining Series..."
seriesResultxx = SHSeries(np.cos(thetaVals[15]), phiVals[15], Nval, C_nXX)
seriesResultxy = SHSeries(np.cos(thetaVals[15]), phiVals[15], Nval, C_nXY)
seriesResultxz = SHSeries(np.cos(thetaVals[15]), phiVals[15], Nval, C_nXZ)
seriesResultyx = SHSeries(np.cos(thetaVals[15]), phiVals[15], Nval, C_nYX)
seriesResultyy = SHSeries(np.cos(thetaVals[15]), phiVals[15], Nval, C_nYY)
seriesResultyz = SHSeries(np.cos(thetaVals[15]), phiVals[15], Nval, C_nYZ)
seriesResultzx = SHSeries(np.cos(thetaVals[15]), phiVals[15], Nval, C_nZX)
seriesResultzy = SHSeries(np.cos(thetaVals[15]), phiVals[15], Nval, C_nZY)
seriesResultzz = SHSeries(np.cos(thetaVals[15]), phiVals[15], Nval, C_nZZ)
#Inverse Metric
invSeriesResultxx = SHSeries(np.cos(thetaVals[15]), phiVals[15], Nval, InvC_nXX)
invSeriesResultxy = SHSeries(np.cos(thetaVals[15]), phiVals[15], Nval, InvC_nXY)
invSeriesResultxz = SHSeries(np.cos(thetaVals[15]), phiVals[15], Nval, InvC_nXZ)
invSeriesResultyx = SHSeries(np.cos(thetaVals[15]), phiVals[15], Nval, InvC_nYX)
invSeriesResultyy = SHSeries(np.cos(thetaVals[15]), phiVals[15], Nval, InvC_nYY)
invSeriesResultyz = SHSeries(np.cos(thetaVals[15]), phiVals[15], Nval, InvC_nYZ)
invSeriesResultzx = SHSeries(np.cos(thetaVals[15]), phiVals[15], Nval, InvC_nZX)
invSeriesResultzy = SHSeries(np.cos(thetaVals[15]), phiVals[15], Nval, InvC_nZY)
invSeriesResultzz = SHSeries(np.cos(thetaVals[15]), phiVals[15], Nval, InvC_nZZ)

seriesResult = np.zeros(np.shape(metricField(0,0)))
seriesResult[0,0] = np.real(seriesResultxx)
seriesResult[0,1] = np.real(seriesResultxy)
seriesResult[0,2] = np.real(seriesResultxz)
seriesResult[1,0] = np.real(seriesResultyx)
seriesResult[1,1] = np.real(seriesResultyy)
seriesResult[1,2] = np.real(seriesResultyz)
seriesResult[2,0] = np.real(seriesResultzx)
seriesResult[2,1] = np.real(seriesResultzy)
seriesResult[2,2] = np.real(seriesResultzz)

invSeriesResult = np.zeros(np.shape(metricField(0,0)))
invSeriesResult[0,0] = np.real(invSeriesResultxx)
invSeriesResult[0,1] = np.real(invSeriesResultxy)
invSeriesResult[0,2] = np.real(invSeriesResultxz)
invSeriesResult[1,0] = np.real(invSeriesResultyx)
invSeriesResult[1,1] = np.real(invSeriesResultyy)
invSeriesResult[1,2] = np.real(invSeriesResultyz)
invSeriesResult[2,0] = np.real(invSeriesResultzx)
invSeriesResult[2,1] = np.real(invSeriesResultzy)
invSeriesResult[2,2] = np.real(invSeriesResultzz)
print "Series determined, printing results..."

print 'Projection Operator: ', projOperator(thetaVals[15],phiVals[15])
print 'Metric Projection XX: ', metricProj(thetaVals[15],phiVals[15],identity,00)
print 'Metric Projection XY: ', metricProj(thetaVals[15],phiVals[15],identity,01)
print 'Metric Projection XZ: ', metricProj(thetaVals[15],phiVals[15],identity,02)
print 'Metric Projection YX: ', metricProj(thetaVals[15],phiVals[15],identity,10)
print 'Metric Projection YY: ', metricProj(thetaVals[15],phiVals[15],identity,11)
print 'Metric Projection YZ: ', metricProj(thetaVals[15],phiVals[15],identity,12)
print 'Metric Projection ZX: ', metricProj(thetaVals[15],phiVals[15],identity,20)
print 'Metric Projection ZY: ', metricProj(thetaVals[15],phiVals[15],identity,21)
print 'Metric Projection ZZ: ', metricProj(thetaVals[15],phiVals[15],identity,22)
#print 'SeriesResult*SeriesResult:', np.matmul(seriesResult,seriesResult)
print 'SeriesResult*InverseSeriesResult:', np.matmul(invSeriesResult,seriesResult)

#Print Results
print "Spherical Harmonics Series XX Coefficients:", np.real(C_nXX)
print "Spherical Harmonics Series XY Coefficients:", np.real(C_nXY)
print "Spherical Harmonics Series XZ Coefficients:", np.real(C_nXZ)
print "Spherical Harmonics Series YX Coefficients:", np.real(C_nYX)
print "Spherical Harmonics Series YY Coefficients:", np.real(C_nYY)
print "Spherical Harmonics Series YZ Coefficients:", np.real(C_nYZ)
print "Spherical Harmonics Series ZX Coefficients:", np.real(C_nZX)
print "Spherical Harmonics Series ZY Coefficients:", np.real(C_nZY)
print "Spherical Harmonics Series ZZ Coefficients:", np.real(C_nZZ)
#Inverse Series
print "Inverse Spherical Harmonics Series XX Coefficients:", np.real(InvC_nXX)
print "Inverse Spherical Harmonics Series XY Coefficients:", np.real(InvC_nXY)
print "Inverse Spherical Harmonics Series XZ Coefficients:", np.real(InvC_nXZ)
print "Inverse Spherical Harmonics Series YX Coefficients:", np.real(InvC_nYX)
print "Inverse Spherical Harmonics Series YY Coefficients:", np.real(InvC_nYY)
print "Inverse Spherical Harmonics Series YZ Coefficients:", np.real(InvC_nYZ)
print "Inverse Spherical Harmonics Series ZX Coefficients:", np.real(InvC_nZX)
print "Inverse Spherical Harmonics Series ZY Coefficients:", np.real(InvC_nZY)
print "Inverse Spherical Harmonics Series ZZ Coefficients:", np.real(InvC_nZZ)
#Check Coefficients
print "Checking Values of XX Coefficients:", checkCoeffxx
print "Checking Values of XY Coefficients:", checkCoeffxy
print "Checking Values of XZ Coefficients:", checkCoeffxz
print "Checking Values of YX Coefficients:", checkCoeffyx
print "Checking Values of YY Coefficients:", checkCoeffyy
print "Checking Values of YZ Coefficients:", checkCoeffyz
print "Checking Values of ZX Coefficients:", checkCoeffzx
print "Checking Values of ZY Coefficients:", checkCoeffzy
print "Checking Values of ZZ Coefficients:", checkCoeffzz
#Inverse Check
print "Checking Inverse Values of XX Coefficients:", invCheckCoeffxx
print "Checking Inverse Values of XY Coefficients:", invCheckCoeffxy
print "Checking Inverse Values of XZ Coefficients:", invCheckCoeffxz
print "Checking Inverse Values of YX Coefficients:", invCheckCoeffyx
print "Checking Inverse Values of YY Coefficients:", invCheckCoeffyy
print "Checking Inverse Values of YZ Coefficients:", invCheckCoeffyz
print "Checking Inverse Values of ZX Coefficients:", invCheckCoeffzx
print "Checking Inverse Values of ZY Coefficients:", invCheckCoeffzy
print "Checking Inverse Values of ZZ Coefficients:", invCheckCoeffzz
#Print Error
print "Error in XX:", errorxx
print "Error in XY:", errorxy
print "Error in XZ:", errorxz
print "Error in YX:", erroryx
print "Error in YY:", erroryy
print "Error in YZ:", erroryz
print "Error in ZX:", errorzx
print "Error in ZY:", errorzy
print "Error in ZZ:", errorzz
#Inverse Error
print "Inverse Error in XX:", invErrorxx
print "Inverse Error in XY:", invErrorxy
print "Inverse Error in XZ:", invErrorxz
print "Inverse Error in YX:", invErroryx
print "Inverse Error in YY:", invErroryy
print "Inverse Error in YZ:", invErroryz
print "Inverse Error in ZX:", invErrorzx
print "Inverse Error in ZY:", invErrorzy
print "Inverse Error in ZZ:", invErrorzz

elapsedTime = time.time() - t
print "Elapsed Time (s):", elapsedTime

#*****************************Representing Vectors**********************************
#t = time.time()
# print "Finding coefficients..."
# C_nx = findCartCoeff(Nval, vecProj, intN, vecField, 0)
# C_ny = findCartCoeff(Nval, vecProj, intN, vecField, 1)
# C_nz = findCartCoeff(Nval, vecProj, intN, vecField, 2)
# print "Coefficients Found!"
#
# checkCoeffx = []
# for check in range(len(C_nx)):
#     if abs(C_nx[check]) > 1e-6:
#         checkCoeffx.append(1)
#     else:
#         checkCoeffx.append(0)
# checkCoeffy = []
# for check in range(len(C_ny)):
#     if abs(C_ny[check]) > 1e-6:
#         checkCoeffy.append(1)
#     else:
#         checkCoeffy.append(0)
# checkCoeffz = []
# for check in range(len(C_nz)):
#     if abs(C_nz[check]) > 1e-6:
#         checkCoeffz.append(1)
#     else:
#         checkCoeffz.append(0)
#
# #List L2 error for each N-value.
# print "Calculating Error List..."
# errorListx = calcCartErrorList(C_nx, Nval, vecProj, intN, vecField, 0)
# print "Error for X-Component Done!"
# errorListy = calcCartErrorList(C_ny, Nval, vecProj, intN, vecField, 1)
# print "Error for Y-Component Done!"
# errorListz = calcCartErrorList(C_nz, Nval, vecProj, intN, vecField, 2)
# print "Error for Z-Component Done!"
#
# errorx = errorListx[len(errorListx)-1]
# errory = errorListy[len(errorListy)-1]
# errorz = errorListz[len(errorListz)-1]
# print "Errors Calculated!"
#
# print "Determining Series..."
# seriesResultx = SHSeries(np.cos(theta_mesh), phi_mesh, Nval, C_nx)
# seriesResulty = SHSeries(np.cos(theta_mesh), phi_mesh, Nval, C_ny)
# seriesResultz = SHSeries(np.cos(theta_mesh), phi_mesh, Nval, C_nz)
# seriesResult = [seriesResultx, seriesResulty, seriesResultz]
# print "Series determined, printing results..."
#
# #Print Results
# print "Spherical Harmonics Series X Coefficients:", np.real(C_nx)
# print "Spherical Harmonics Series Y Coefficients:", np.real(C_ny)
# print "Spherical Harmonics Series Z Coefficients:", np.real(C_nz)
# print "Checking Values of X Coefficients:", checkCoeffx
# print "Checking Values of Y Coefficients:", checkCoeffy
# print "Checking Values of Z Coefficients:", checkCoeffz
# print "Error in X:", errorx
# print "Error in Y:", errory
# print "Error in Z:", errorz
#
# elapsedTime = time.time() - t
# print "Elapsed Time (s):", elapsedTime
# print "Plotting Results..."
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.quiver(X_mesh[::10,::10], Y_mesh[::10,::10] ,Z_mesh[::10,::10], np.real(seriesResultx)[::10,::10], np.real(seriesResulty)[::10,::10], np.real(seriesResultz)[::10,::10])
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')
# plt.show()
#
# print "DONE!"

#elapsedTime = time.time() - t
#print "Elapsed Time (s):", elapsedTime























# *******************UNUSED PLOTTING CODE*******************************
# #Scatter plot the derivative L2 error versus N
# plt.figure()
# plt.scatter(coeffNum, np.log10(derivErrorList))
# #plt.yscale('log')
# plt.xlabel('N-Value')
# plt.ylabel('Log_10 of Derivative L2 Error')
# plt.grid()
# plt.title('Derivative L2 Error for Different N-Values')
#
# #Plot Results
# plt.figure()
# plt.plot(xvals, seriesResult, 'r', label='Legendre Series')
# plt.plot(xvals, DesiredFunction(xvals), 'b--', label='Desired Function')
# plt.grid()
# plt.legend()
# plt.xlabel('X-Values')
# plt.ylabel('Y-Values')
# plt.title('Representing Functions Using Legendre Polynomials')
#
# #Plot Deriv Results
# plt.figure()
# plt.plot(xvals, derivSeriesResult, 'r', label='Derivative Legendre Series')
# plt.plot(xvals, DerivFunction(xvals), 'b--', label='Derivative Function')
# plt.grid()
# plt.legend()
# plt.xlabel('X-Values')
# plt.ylabel('Y-Values')
# plt.title('Representing Derivatives of Functions Using Legendre Polynomials')
#
# #Plot Phi Results
# plt.figure()
# plt.plot(xvals, phiSeries, 'r', label='Phi Series')
# plt.plot(xvals, phi(xvals), 'b--', label='Analytic Phi')
# plt.grid()
# plt.legend()
# plt.xlabel('X-Values')
# plt.ylabel('Y-Values')
# plt.title('Determining Phi Using Legendre Polynomials')
#
# plt.show()