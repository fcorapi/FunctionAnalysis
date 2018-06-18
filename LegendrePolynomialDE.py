#LegendrePolynomialDerivative
#This script takes a given function and calculates its derivative using a Legendre polynomial recursion relation.
#Frank Corapi (fcorapi@uwaterloo.ca)
#Last Modified: 06/08/2018

#Import Directories
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre
from scipy.integrate import quad

#X-Values
xvals = np.linspace(-1.0, 1.0, 1000)

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

#Function to be integrated to determine Legendre coefficients
def integrand(x,n, fn):
    value = Legendre(x,n)*fn(x)
    return value

#Empty list that will contain Legendre series coefficients
C_n = []

#Integrate to determine Legendre series coefficients
Nval = 60 #Number of coefficients
for n in range(0, Nval):
    integralValue = quad(integrand, -1.0, 1.0, args=(n, DesiredFunction,))
    cval = ((2.0*n+1)/2.0)*integralValue[0]
    C_n.append(cval)

#LegendreSeries
def LegendreSeries(x, N, coeff):
    series = 0
    if N > len(C_n):
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

#print np.matmul(DerivMatrix(10), DerivMatrix(10))

#Create a list for the coefficients of the series representing the derivative of the function
Cprime_n = np.matmul(DerivMatrix(len(C_n)), np.matmul(DerivMatrix(len(C_n)), C_n))

#print len(C_n)
#print DerivMatrix(len(C_n))
#print Cprime_n

#MAY REMOVE
#DerivLegendreSeries
# def DerivLegendreSeries(x, N):
#     series = 0
#     if N > len(C_n):
#         print "Error"
#         return 0
#     else:
#         for loop in range(0, N):
#             series = series + Cprime_n[loop]*Legendre(x, loop)
#     return series

#Function to integrate over to find error in the Legendre Series
def L2ErrorFunction(x, N, coeff, fn):
    errVal = abs(LegendreSeries(x, N, coeff) - fn(x))**2
    return errVal

#MAY REMOVE
# def DerivL2ErrorFunction(x, N):
#     errVal = abs(DerivLegendreSeries(x, N) - DerivFunction(x))**2
#     return errVal

#Make empty list to contain the L2 error for each N-value, as well as a list of the N-values.
errorList = []
derivErrorList = []
coeffNum = np.linspace(0,Nval-1,Nval)

#Loops over every N value up to a maximum, and calculates the L2 error, and plots the square of the error as a function
#of x.
for maxN in range(1, len(C_n)+1):
    error = quad(L2ErrorFunction, -1.0, 1.0, args=(maxN, C_n, DesiredFunction, ))
    errorList.append(np.sqrt(error[0]))

    derivError = quad(L2ErrorFunction, -1.0, 1.0, args=(maxN, Cprime_n, DerivFunction, ))
    derivErrorList.append(np.sqrt(derivError[0]))

    #plt.plot(xvals, L2ErrorFunction(xvals, maxN), label = "N = " + str(maxN-1))
    #plt.legend()
    #plt.show()

#Error plot properties
#plt.legend()
#plt.title("Square of the Error for Different N-Values")

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

#Error and Series Solution
error = quad(L2ErrorFunction, -1.0, 1.0, args=(len(C_n), C_n, DesiredFunction, ))
error = np.sqrt(error[0])
derivError = quad(L2ErrorFunction, -1.0, 1.0, args=(len(Cprime_n), Cprime_n, DerivFunction, ))
derivError = np.sqrt(derivError[0])
seriesResult = LegendreSeries(xvals, len(C_n), C_n)
derivSeriesResult = LegendreSeries(xvals, len(Cprime_n), Cprime_n)

#Print Results
print "Legendre Series Coefficients:", C_n
print "Error:", error
print "Derivative Legendre Series Coefficients", Cprime_n
print "Derivative Error:", derivError

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
plt.show()


