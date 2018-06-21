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
    val = -20.0*(np.exp(-2.0*(x**2)) - np.exp(-2.0))
    return val

#Mass Density Function Example (rho)
def rho(x):
    val = 80.0*np.exp(-2.0*(x**2))*(1.0 - 4.0*(x**2))
    return val

#Function to be integrated to determine Legendre coefficients
def integrand(x,n, fn):
    value = Legendre(x,n)*fn(x)
    return value

def findCoeff(Nval, fn):
    coeffList = []
    # Integrate to determine Legendre series coefficients
    for n in range(0, Nval):
        integralValue = quad(integrand, -1.0, 1.0, args=(n, fn,))
        cval = ((2.0*n+1)/2.0)*integralValue[0]
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
def calcErrorList(coeff, fn):
    errList = []
    for maxN in range(1, len(coeff) + 1):
        error = quad(L2ErrorFunction, -1.0, 1.0, args=(maxN, coeff, fn,))
        errList.append(np.sqrt(error[0]))
    return errList
#*******************************END OF FUNCTIONS*************************************


Nval = 60 #Number of coefficients
xvals = np.linspace(-1.0, 1.0, 1000) #X-Values
coeffNum = np.linspace(0,Nval-1,Nval) #List of N-values

C_n = findCoeff(Nval, DesiredFunction) #Coefficients of Desired Function
Cprime_n = calcDeriv(2, C_n) #Coefficients of the derivative of the function

#List L2 error for each N-value.
errorList = calcErrorList(C_n, DesiredFunction)
derivErrorList = calcErrorList(Cprime_n, DerivFunction)

#***************Solving an ODE***************

rho_n = findCoeff(Nval,rho)
phi_n = np.linalg.solve(LMatrix(Nval), rho_n)
phiErrorList = calcErrorList(phi_n, phi)

#Error and Series Solution for Phi
phiError = quad(L2ErrorFunction, -1.0, 1.0, args=(len(phi_n), phi_n, phi, ))
phiError = np.sqrt(phiError[0])
phiSeries = LegendreSeries(xvals, len(phi_n), phi_n)

#***************************

#Error and Series Solution
error = quad(L2ErrorFunction, -1.0, 1.0, args=(len(C_n), C_n, DesiredFunction, ))
error = np.sqrt(error[0])
derivError = quad(L2ErrorFunction, -1.0, 1.0, args=(len(Cprime_n), Cprime_n, DerivFunction, ))
derivError = np.sqrt(derivError[0])
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