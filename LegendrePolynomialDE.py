#LegendrePolynomialDE
#This script takes a given function and calculates its derivative using a Legendre polynomial recursion relation,
#and will work to solve DEs using similar methods.
#Frank Corapi (fcorapi@uwaterloo.ca)
#Last Modified: 06/20/2018

#Import Directories
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre
from scipy.integrate import quad
from numpy.polynomial.legendre import legroots
#***********************************FUNCTION DEFINITIONS*************************************

#Define Legendre Function
def Legendre(x,n):
    leg = legendre(n)
    P_n = leg(x)
    return P_n

#Function that we want to represent as a Legendre Series
#TO BE MODIFIED BY USER DEPENDING ON WHICH FUNCTION IS WANTED
def DesiredFunction(x):
    val = np.sin(20*x)
    return val

#Derivative of the desired function
def DerivFunction(x):
    val = 20.0*(-20.0)*np.sin(20*x)
    return val

#Analytic result for the phi function
def phi(x):
    val = np.cos((21*np.pi/2.0)*x)
    return val

#Mass Density Function Example (rho)
def rho(x):
    val = -1.0*((21*np.pi/2.0)**2)*np.cos((21*np.pi/2.0)*x)
    return val

#Function to be integrated to determine Legendre coefficients
def integrand(x,n, fn):
    value = Legendre(x,n)*fn(x)
    return value

def findCoeff(Nval, fn, intTerms):
    coeffList = []
    # Integrate to determine Legendre series coefficients
    for n in range(0, Nval):
        integralValue = GL_Quad(integrand, -1.0, 1.0, intTerms, args=(n, fn,))
        cval = ((2.0*n+1)/2.0)*integralValue
        coeffList.append(cval)
    return coeffList

#LegendreSeries
def LegendreSeries(x, N, coeff):
    series = 0
    if N > len(coeff):
        print "Error"
        return 0
    else:
        for loop in range(0, N):
            series = series + coeff[loop]*Legendre(x, loop)
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
def L2ErrorFunction(x, N, coeff, fn):
    errVal = abs(LegendreSeries(x, N, coeff) - fn(x))**2
    return errVal

#Loops over every N value up to a maximum, and calculates the L2 error.
def calcErrorList(coeff, fn, intTerms):
    errList = []
    for maxN in range(1, len(coeff) + 1):
        err = GL_Quad(L2ErrorFunction, -1.0, 1.0, intTerms, args=(maxN, coeff, fn,))
        errList.append(np.sqrt(err))
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

#*******************************END OF FUNCTIONS*************************************


Nval = 60 #Number of coefficients
intN = Nval + 25 #Number of terms in Gauss-Legendre integration
xvals = np.linspace(-1.0, 1.0, 1000) #X-Values
coeffNum = np.linspace(0,Nval-1,Nval) #List of N-values

C_n = findCoeff(Nval, DesiredFunction, intN) #Coefficients of Desired Function
Cprime_n = calcDeriv(2, C_n) #Coefficients of the derivative of the function

#List L2 error for each N-value.
errorList = calcErrorList(C_n, DesiredFunction, intN)
derivErrorList = calcErrorList(Cprime_n, DerivFunction, intN)

#***************Solving an ODE***************

rho_n = findCoeff(Nval, rho, intN)
phi_n = np.linalg.solve(LMatrix(Nval), rho_n)
phiErrorList = calcErrorList(phi_n, phi, intN)

#Error and Series Solution for Phi
phiError = GL_Quad(L2ErrorFunction, -1.0, 1.0, Nval, args=(len(phi_n), phi_n, phi, ))
phiError = np.sqrt(phiError)
phiSeries = LegendreSeries(xvals, len(phi_n), phi_n)

#***************************

#Error and Series Solution
error = GL_Quad(L2ErrorFunction, -1.0, 1.0, intN, args=(len(C_n), C_n, DesiredFunction, ))
error = np.sqrt(error)
derivError = GL_Quad(L2ErrorFunction, -1.0, 1.0, intN, args=(len(Cprime_n), Cprime_n, DerivFunction, ))
derivError = np.sqrt(derivError)
seriesResult = LegendreSeries(xvals, len(C_n), C_n)
derivSeriesResult = LegendreSeries(xvals, len(Cprime_n), Cprime_n)

#Print Results
#print "Legendre Series Coefficients:", C_n
print "Error:", error
#print "Derivative Legendre Series Coefficients", Cprime_n
print "Derivative Error:", derivError

#print "Phi Coefficients:", phi_n
print "Phi Error", phiError

#Scatter plot the L2 error versus N
plt.figure()
plt.scatter(coeffNum, np.log10(errorList))
#plt.yscale('log')
plt.xlabel('N-Value')
plt.ylabel('Log_10 of L2 Error')
plt.grid()
plt.title('L2 Error for Different N-Values')

#Scatter plot the derivative L2 error versus N
plt.figure()
plt.scatter(coeffNum, np.log10(derivErrorList))
#plt.yscale('log')
plt.xlabel('N-Value')
plt.ylabel('Log_10 of Derivative L2 Error')
plt.grid()
plt.title('Derivative L2 Error for Different N-Values')

#Scatter plot the phi L2 error versus N
plt.figure()
plt.scatter(coeffNum, np.log10(phiErrorList))
#plt.yscale('log')
plt.xlabel('N-Value')
plt.ylabel('Log_10 of Phi L2 Error')
plt.grid()
plt.title('Phi L2 Error for Different N-Values')

#Plot Results
plt.figure()
plt.plot(xvals, seriesResult, 'r', label='Legendre Series')
plt.plot(xvals, DesiredFunction(xvals), 'b--', label='Desired Function')
plt.grid()
plt.legend()
plt.xlabel('X-Values')
plt.ylabel('Y-Values')
plt.title('Representing Functions Using Legendre Polynomials')

#Plot Deriv Results
plt.figure()
plt.plot(xvals, derivSeriesResult, 'r', label='Derivative Legendre Series')
plt.plot(xvals, DerivFunction(xvals), 'b--', label='Derivative Function')
plt.grid()
plt.legend()
plt.xlabel('X-Values')
plt.ylabel('Y-Values')
plt.title('Representing Derivatives of Functions Using Legendre Polynomials')

#Plot Phi Results
plt.figure()
plt.plot(xvals, phiSeries, 'r', label='Phi Series')
plt.plot(xvals, phi(xvals), 'b--', label='Analytic Phi')
plt.grid()
plt.legend()
plt.xlabel('X-Values')
plt.ylabel('Y-Values')
plt.title('Determining Phi Using Legendre Polynomials')

plt.show()