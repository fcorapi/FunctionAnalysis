#SphericalHarmonicsDE
#This script takes a given function and represents it as a series of Spherical Harmonics,
#and will eventually work to solve DEs.
#Frank Corapi (fcorapi@uwaterloo.ca)
#Last Modified: 08/02/2018

#Import Directories
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre, sph_harm
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
#NEEDS TO BE MODIFIED
def DerivFunction(x):
    val = 20.0*(-20.0)*np.sin(20*x)
    return val

#Analytic result for the phi function
def phi(theta, phi):
    val = 0
    for l in range(0, 25):
        for m in np.arange(-l, l, 2):
            val = sph_harm(m, l, phi, theta) + val
    return np.real(val)

#Mass Density Function Example (rho)
def rho(theta, phi):
    val = 0
    for l in range(1, 25):
        for m in np.arange(-l, l, 2):
            val = -l*(l+1)*sph_harm(m, l, phi, theta) + val
    return np.real(val)

#Function to be integrated to determine Legendre coefficients
def integrand(z, phi, n, m, fn):
    value = np.conj(sph_harm(m, n, phi, np.arccos(z)))*fn(np.arccos(z), phi)
    return value

def findCoeff(Nval, fn, intTerms):
    coeffList = []
    # Integrate to determine Legendre series coefficients
    for n in range(0, Nval):
        for m in range(-n, n+1):
            integralValue = GL_Quad_2D(integrand, -1.0, 1.0, 0, 2*np.pi, intTerms, args=(n, m, fn,))
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
        err = GL_Quad_2D(L2ErrorFunction, -1.0, 1.0, 0, 2*np.pi, intTerms, args=(maxN, coeff, fn,))
        errList.append(np.sqrt(err))
        print "Error for N = ", maxN, " completed."
    return errList

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
    phiVals = np.linspace(lowPhi, upPhi-deltaPhi, Npoints)

    # Zpoints = 1000
    # deltaZ = (upZ-lowZ)/Zpoints
    # Zvals = np.linspace(lowZ, upZ, Zpoints+1)
    # ztracker = 0

    value = 0
    for z_i in roots:
        phiValue = 0
        for phi in phiVals:
                phiValue = deltaPhi * psi4(z_i) * integrand(((upZ - lowZ) / 2.0) * z_i + ((upZ + lowZ) / 2.0), phi, *args) + phiValue
        value = weight(z_i)*phiValue + value

    value = ((upZ-lowZ)/2.0)*value
    return value

#*********************************************VECTOR SPHERICAL HARMONICS*************************************
#Define conformal factor
def psi4(z):
    factor = 1 - z**2
    return factor

#Partial Phi derivative of a spherical harmonic
def phiDerivSH(M, N, phi, theta):
    eps = 1e-5
    deriv = (sph_harm(M, N, phi + 0.5*eps, theta) - sph_harm(M, N, phi - 0.5*eps, theta))/eps
    return deriv

#Partial Theta derivative of a spherical harmonic
def thetaDerivSH(M, N, phi, theta):
    eps = 1e-5
    deriv = (sph_harm(M, N, phi, theta + 0.5*eps) - sph_harm(M, N, phi, theta - 0.5*eps))/eps
    return deriv

def VecDesiredFunction(theta,phi,kind):
    A = 2 * np.sqrt((np.pi) / 3)
    if kind == 'polar':
        val = [-A*thetaDerivSH(0, 1, phi, theta), -A*phiDerivSH(0, 1, phi, theta)]
    elif kind == 'axial':
        val = [A*phiDerivSH(0, 1, phi, theta)/np.sin(theta), -A*np.sin(theta) * thetaDerivSH(0, 1, phi, theta)]
    return val

#Representing Phi1 vector field from Korzynski paper using vector SH
def Phi1(theta, phi, kind):
    if kind == 'axial':
        A = 2*np.sqrt((2*np.pi)/3)
        val = np.subtract([-0.5 * A * phiDerivSH(1, 1, phi, theta)/np.sin(theta), 0.5 * A * np.sin(theta) * thetaDerivSH(1, 1, phi, theta)],
                          [-0.5 * A * phiDerivSH(-1, 1, phi, theta)/np.sin(theta), 0.5 * A * np.sin(theta) * thetaDerivSH(-1, 1, phi, theta)])
    elif kind == 'polar':
        val = [0,0]
        print "Error, Phi1 is not a polar vector!"
    return val

#Representing Phi2 vector field from Korzynski paper using vector SH
def Phi2(theta, phi, kind):
    if kind == 'axial':
        A = 2 * np.sqrt((2 * np.pi) / 3)
        val = np.add([0.5j * A * phiDerivSH(1, 1, phi, theta)/np.sin(theta), -0.5j * A * np.sin(theta) * thetaDerivSH(1, 1, phi, theta)],
                     [0.5j * A * phiDerivSH(-1, 1, phi, theta)/np.sin(theta), -0.5j * A * np.sin(theta) * thetaDerivSH(-1, 1, phi, theta)])
    elif kind == 'polar':
        val = [0,0]
        print "Error, Phi2 is not a polar vector!"
    return val

#Representing Phi3 vector field from Korzynski paper using vector SH
def Phi3(theta, phi, kind):
    if kind == 'axial':
        A = 2 * np.sqrt((np.pi) / 3)
        val = [A * phiDerivSH(0, 1, phi, theta)/np.sin(theta), -A * np.sin(theta) * thetaDerivSH(0, 1, phi, theta)]
    elif kind == 'polar':
        val = [0,0]
        print "Error, Phi3 is not a polar vector!"
    return val

#Representing Xi1 vector field from Korzynski paper using vector SH
def Xi1(theta, phi, kind):
    if kind == 'polar':
        A = 2 * np.sqrt((2 * np.pi) / 3)
        val = np.subtract([0.5 * A * thetaDerivSH(1, 1, phi, theta), 0.5 * A * phiDerivSH(1, 1, phi, theta)],
                          [0.5 * A * thetaDerivSH(-1, 1, phi, theta), 0.5 * A * phiDerivSH(-1, 1, phi, theta)])
    elif kind == 'axial':
        val = [0,0]
        print "Error, Xi1 is not an axial vector!"
    return val

#Representing Xi2 vector field from Korzynski paper using vector SH
def Xi2(theta, phi, kind):
    if kind == 'polar':
        A = 2 * np.sqrt((2 * np.pi) / 3)
        val = np.add([-0.5j * A * thetaDerivSH(1, 1, phi, theta), -0.5j * A * phiDerivSH(1, 1, phi, theta)],
                     [-0.5j * A * thetaDerivSH(-1, 1, phi, theta), -0.5j * A * phiDerivSH(-1, 1, phi, theta)])
    elif kind == 'axial':
        val = [0,0]
        print "Error, Xi2 is not an axial vector!"
    return val

#Representing Xi3 vector field from Korzynski paper using vector SH
def Xi3(theta, phi, kind):
    if kind == 'polar':
        A = 2 * np.sqrt((np.pi) / 3)
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
            value = value + (1/psi4(z)) * q_inv[loop1, loop2]*np.conj(vectorSH(m, n, phi, np.arccos(z), kind))[loop1]*fn(np.arccos(z), phi, kind)[loop2]
    #value = np.dot(np.conj(vectorSH(m, n, phi, np.arccos(z), kind)), fn(np.arccos(z), phi, kind))
    return value

def findVecCoeff(Nval, fn, intTerms, kind):
    coeffList = []
    # Integrate to determine Legendre series coefficients
    for n in range(0, Nval):
        for m in range(-n, n+1):
            integralValue = GL_Quad_2D(vecIntegrand, -1.0, 1.0, 0, 2*np.pi, intTerms, args=(n, m, fn, kind,))
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

#Function to integrate over to find error in the Legendre Series
def L2VecErrorFunction(z, phi, N, coeff, fn, kind):
    # Define the spherical metric
    q_inv = np.zeros((2, 2))
    q_inv[0, 0] = 1
    q_inv[1, 1] = 1 / (np.sin(np.arccos(z))**2)

    diff = VecSHSeries(z, phi, N, coeff, kind) - fn(np.arccos(z), phi, kind)
    errVal = 0

    for loop1 in range(0,2):
        for loop2 in range(0,2):
            errVal = errVal + (1/psi4(z))*q_inv[loop1, loop2]*np.conj(diff)[loop1]*diff[loop2]
    #errVal = abs(np.dot(np.conj(VecSHSeries(z, phi, N, coeff, kind) - fn(np.arccos(z), phi, kind)), VecSHSeries(z, phi, N, coeff, kind) - fn(np.arccos(z), phi, kind)))
    return abs(errVal)

#Loops over every N value up to a maximum, and calculates the L2 error.
def calcVecErrorList(coeff, Nval, fn, intTerms, kind):
    errList = []
    for maxN in range(1, Nval + 1):
        err = GL_Quad_2D(L2VecErrorFunction, -1.0, 1.0, 0, 2*np.pi, intTerms, args=(maxN, coeff, fn, kind,))
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
            value = value + (1/psi4(z)) * q_inv[loop1, loop2] * w(np.arccos(z), phi, N, wcoeff, wKind)[loop1] * vecFn(np.arccos(z), phi, fnKind)[loop2]
    # value = np.dot(np.conj(vectorSH(m, n, phi, np.arccos(z), kind)), fn(np.arccos(z), phi, kind))
    return value

def calculateJK(w, vecFn, N, intTerms, wcoeff, wKind, fnKind):

    integralValue = GL_Quad_2D(JKIntegrand, -1.0, 1.0, 0, 2 * np.pi, intTerms, args=(w, vecFn, N, wcoeff, wKind, fnKind,))

    return integralValue
#*******************************END OF FUNCTIONS*************************************

Nval = 2 #Number of coefficients
intN = 3*Nval #Number of terms in Gauss-Legendre integration
thetaVals = np.linspace(0, np.pi, 100) + 1e-5#Theta-Values
phiVals = np.linspace(0, 2*np.pi, 100) + 1e-5 #Phi-Values
theta_mesh, phi_mesh = np.meshgrid(thetaVals, phiVals) #Make a mesh grid
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
t = time.time()
vecKind = 'axial' #The Kind has to be 'polar' or 'axial'

print "Finding coefficients..."
C_n = findVecCoeff(Nval, Phi3, intN, vecKind) #Coefficients of Desired Function
print "Coefficients Found!"
# Cprime_n = calcDeriv(2, C_n) #Coefficients of the derivative of the function

checkCoeff = []
for check in range(len(C_n)):
    if abs(C_n[check]) > 1e-6:
        checkCoeff.append(1)
    else:
        checkCoeff.append(0)

#List L2 error for each N-value.
print "Calculating Error List..."
errorList = calcVecErrorList(C_n, Nval, VecDesiredFunction, intN, vecKind)
# derivErrorList = calcErrorList(Cprime_n, Nval, DerivFunction, intN)
error = errorList[len(errorList)-1]
# derivError = derivErrorList[len(derivErrorList)-1]
print "Errors Calculated!"

print "Determining Series..."
seriesResult = VecSHSeries(np.cos(theta_mesh), phi_mesh, Nval, C_n, vecKind)
print np.imag(seriesResult)
print "Series determined, plotting results..."
# derivSeriesResult = SHSeries(thetaVals, phiVals, Nval, Cprime_n)

#Print Results
print "Spherical Harmonics Series Coefficients:", C_n
print "Checking Values of Coeffecients:", checkCoeff
print "Error:", error
# print "Derivative Legendre Series Coefficients", Cprime_n
# print "Derivative Error:", derivError


#Calculating the J and K values from the Korzynski paper.
J1 = (-1/(8*np.pi*G))*calculateJK(omega, Phi1, Nval, intN, C_n, vecKind, 'axial')
J2 = (-1/(8*np.pi*G))*calculateJK(omega, Phi2, Nval, intN, C_n, vecKind, 'axial')
J3 = (-1/(8*np.pi*G))*calculateJK(omega, Phi3, Nval, intN, C_n, vecKind, 'axial')
K1 = (-1/(8*np.pi*G))*calculateJK(omega, Xi1, Nval, intN, C_n, vecKind, 'polar')
K2 = (-1/(8*np.pi*G))*calculateJK(omega, Xi2, Nval, intN, C_n, vecKind, 'polar')
K3 = (-1/(8*np.pi*G))*calculateJK(omega, Xi3, Nval, intN, C_n, vecKind, 'polar')

#Calculate the values of the invariants A and B from the  Korzynski paper
invariantA = (J1*J1 + J2*J2 + J3*J3) - (K1*K1 + K2*K2 + K3*K3)
invariantB = K1*J1 + K2*J2 + K3*J3

#Final value for the Angular Momentum
J = np.sqrt((invariantA + np.sqrt(invariantA**2 + 4*(invariantB**2)))/2)

#Calculate properties of the sphere
area = GL_Quad_2D(oneFunction, -1, 1, 0, 2*np.pi, intN, args=())
arealRadius = np.sqrt(area/(4*np.pi))
irrMass = arealRadius/2
mass = np.sqrt(irrMass**2 + J**2/(4*(irrMass**2)))
spin = J/(mass**2)

#Printing the results of the calculated J and K values.
print "The omega one-form is equal to the co-vector phi_3."
print "J_1 = ", J1
print "J_2 = ", J2
print "J_3 = ", J3
print "K_1 = ", K1
print "K_2 = ", K2
print "K_3 = ", K3

print "A = ", invariantA
print "B = ", invariantB
print "J = ", J

print "Properties of the Surface:"
print "Area: ", area
print "Areal Radius: ", arealRadius
print "Irreducible Mass: ", irrMass
print "Mass: ", mass
print "Spin (a): ", spin

elapsedTime = time.time() - t
print "Elapsed Time (s):", elapsedTime

#Scatter plot the L2 error versus N
plt.figure()
plt.scatter(coeffNum, np.log10(errorList))
#plt.yscale('log')
plt.xlabel('N-Value')
plt.ylabel('Log_10 of L2 Error')
plt.grid()
plt.title('L2 Error for Different N-Values $(\\phi_3)$')
plt.show()

#Plotting the vector fields using quiver
plt.figure()
plt.quiver(phi_mesh[::4,::4], theta_mesh[::4,::4], seriesResult[1,::4,::4], -seriesResult[0,::4,::4])
plt.xlabel('$\\phi$-Values')
plt.ylabel('$\\theta$-Values')
plt.gca().invert_yaxis()
plt.title('Vector Spherical Harmonics Series Plot $(\\phi_3)$')
plt.figure()
plt.quiver(phi_mesh[::4,::4], theta_mesh[::4,::4], VecDesiredFunction(theta_mesh, phi_mesh, vecKind)[1][::4,::4],
           -VecDesiredFunction(theta_mesh, phi_mesh, vecKind)[0][::4,::4], color = 'r')
plt.xlabel('$\\phi$-Values')
plt.ylabel('$\\theta$-Values')
plt.gca().invert_yaxis()
plt.title('Vector Desired Function Plot $(\\phi_3)$')
plt.show()


#***************Solving Laplace Equation***************
# t = time.time()
# print "Finding coefficients..."
# rho_n = findCoeff(Nval, rho, intN)
# rho_0 = findCoeff(1, phi, intN)[0]
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