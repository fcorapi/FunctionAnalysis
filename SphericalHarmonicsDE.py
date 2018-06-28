#LegendrePolynomialDE
#This script takes a given function and calculates its derivative using a Legendre polynomial recursion relation,
#and will work to solve DEs using similar methods.
#Frank Corapi (fcorapi@uwaterloo.ca)
#Last Modified: 06/20/2018

#Import Directories
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre, sph_harm
from scipy.integrate import quad
from numpy.polynomial.legendre import legroots
from pylab import imshow
from mpl_toolkits import mplot3d
#***********************************FUNCTION DEFINITIONS*************************************

#Define Legendre Function
def Legendre(x,n):
    leg = legendre(n)
    P_n = leg(x)
    return P_n

#Function that we want to represent as a Legendre Series
#TO BE MODIFIED BY USER DEPENDING ON WHICH FUNCTION IS WANTED
def DesiredFunction(theta,phi):
    val = np.sin(10*theta)*np.cos(phi)
    return val

#Derivative of the desired function
#NEEDS TO BE MODIFIED
def DerivFunction(x):
    val = 20.0*(-20.0)*np.sin(20*x)
    return val

#Analytic result for the phi function
#NEEDS TO BE MODIFIED
def phi(x):
    val = np.cos((21*np.pi/2.0)*x)
    return val

#Mass Density Function Example (rho)
#NEEDS TO BE MODIFIED
def rho(x):
    val = -1.0*((21*np.pi/2.0)**2)*np.cos((21*np.pi/2.0)*x)
    return val

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

def Laplace(Nsize):
    lap = np.zeros((Nsize, Nsize))
    n = 0
    m = 0
    for loopn in range(0, Nsize):
        if m >= 2*n+1:
            m = 0
            n = n + 1
        lap[loopn, loopn] = n*(n+1)
        m = m + 1
    return lap

#print Laplace(18)

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
    Npoints = 30
    deltaPhi = (upPhi-lowPhi)/Npoints
    phiVals = np.linspace(lowPhi, upPhi-deltaPhi, Npoints)

    value = 0
    for z_i in roots:
        phiValue = 0
        for phi in phiVals:
                phiValue = deltaPhi * integrand(((upZ - lowZ) / 2.0) * z_i + ((upZ + lowZ) / 2.0), phi, *args) + phiValue
        value = weight(z_i)*phiValue + value

    value = ((upZ-lowZ)/2.0)*value
    return value

#*******************************END OF FUNCTIONS*************************************


Nval = 25 #Number of coefficients
intN = 2*Nval #Number of terms in Gauss-Legendre integration
thetaVals = np.linspace(0, np.pi, 1000) #Theta-Values
phiVals = np.linspace(0, 2*np.pi, 1000) #Phi-Values
theta_mesh, phi_mesh = np.meshgrid(thetaVals, phiVals) #Make a mesh grid
coeffNum = np.linspace(0,Nval-1,Nval) #List of N-values

C_n = findCoeff(Nval, DesiredFunction, intN) #Coefficients of Desired Function
print "Coefficients Found!"
# Cprime_n = calcDeriv(2, C_n) #Coefficients of the derivative of the function

#List L2 error for each N-value.
print "Calculating Error List..."
errorList = calcErrorList(C_n, Nval, DesiredFunction, intN)
# derivErrorList = calcErrorList(Cprime_n, Nval, DerivFunction, intN)

#***************Solving an ODE***************

# rho_n = findCoeff(Nval, rho, intN)
# phi_n = np.linalg.solve(LMatrix(Nval), rho_n)
# phiErrorList = calcErrorList(phi_n, Nval, phi, intN)
#
# #Error and Series Solution for Phi
# phiError = GL_Quad_2D(L2ErrorFunction, -1.0, 1.0, 0, 2*np.pi, intN, args=(Nval, phi_n, phi, ))
# phiError = np.sqrt(phiError)
# phiSeries = SHSeries(thetaVals, phiVals, Nval, phi_n)

#***************************

#Error and Series Solution
error = GL_Quad_2D(L2ErrorFunction, -1.0, 1.0, 0, 2*np.pi, intN, args=(Nval, C_n, DesiredFunction, ))
error = np.sqrt(error)
# derivError = GL_Quad_2D(L2ErrorFunction, -1.0, 1.0, 0, 2*np.pi, intN, args=(len(Cprime_n), Cprime_n, DerivFunction, ))
# derivError = np.sqrt(derivError)

seriesResult = SHSeries(np.cos(theta_mesh), phi_mesh, Nval, C_n)
# derivSeriesResult = SHSeries(thetaVals, phiVals, Nval, Cprime_n)


ax = plt.axes(projection='3d')
ax.plot_surface(theta_mesh, phi_mesh, np.real(seriesResult), cmap = 'viridis', edgecolor='none')
ax.plot_surface(theta_mesh, phi_mesh, DesiredFunction(theta_mesh, phi_mesh), edgecolor='none')
ax.set_title('Spherical Harmonics Series')
ax.set_xlabel('Theta-Values')
ax.set_ylabel('Phi-Values')
plt.show(ax)

plt.figure()
plt.contourf(theta_mesh, phi_mesh, np.real(seriesResult)-DesiredFunction(theta_mesh, phi_mesh), 30, cmap='hot')
plt.colorbar()
plt.title('Error Plot')
plt.xlabel('Theta-Values')
plt.ylabel('Phi-Values')
plt.show()
#Print Results
print "Spherical Harmonics Series Coefficients:", C_n
print "Error:", error
# print "Derivative Legendre Series Coefficients", Cprime_n
# print "Derivative Error:", derivError

# print "Phi Coefficients:", phi_n
# print "Phi Error", phiError

# #Scatter plot the L2 error versus N
plt.figure()
plt.scatter(coeffNum, np.log10(errorList))
#plt.yscale('log')
plt.xlabel('N-Value')
plt.ylabel('Log_10 of L2 Error')
plt.grid()
plt.title('L2 Error for Different N-Values')
plt.show()
#
# #Scatter plot the derivative L2 error versus N
# plt.figure()
# plt.scatter(coeffNum, np.log10(derivErrorList))
# #plt.yscale('log')
# plt.xlabel('N-Value')
# plt.ylabel('Log_10 of Derivative L2 Error')
# plt.grid()
# plt.title('Derivative L2 Error for Different N-Values')
#
# #Scatter plot the phi L2 error versus N
# plt.figure()
# plt.scatter(coeffNum, np.log10(phiErrorList))
# #plt.yscale('log')
# plt.xlabel('N-Value')
# plt.ylabel('Log_10 of Phi L2 Error')
# plt.grid()
# plt.title('Phi L2 Error for Different N-Values')
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