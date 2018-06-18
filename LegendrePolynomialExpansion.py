#LegendrePolynomialExpansion
#This script takes a given function and represents it as a series of Legendre polynomials by finding the corresponding
#Legendre coefficients.
#Frank Corapi (fcorapi@uwaterloo.ca)
#Last Modified: 05/31/2018

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
def DesiredFuncton(x):
    val = np.sin(30*x)
    return val

#Function to be integrated
def integrand(x,n):
    value = Legendre(x,n)*DesiredFuncton(x)
    return value

#Empty list that will contain Legendre series coefficients
C_n = []

#Integrate to determine Legendre series coefficients
Nval = 71
for n in range(0, Nval):
    integralValue = quad(integrand, -1.0, 1.0, args=(n,))
    cval = ((2.0*n+1)/2.0)*integralValue[0]
    C_n.append(cval)

#LegendreSeries
def LegendreSeries(x, N):
    series = 0
    if N > len(C_n):
        print "Error"
        return 0
    else:
        for loop in range(0, N):
            series = series + C_n[loop]*Legendre(x, loop)
    return series

#Function to integrate over to find error in the Legendre Series
def L2ErrorFunction(x, N):
    errVal = abs(LegendreSeries(x, N) - DesiredFuncton(x))**2
    return errVal

#Make empty list to contain the L2 error for each N-value, as well as a list of the N-values.
errorList = []
coeffNum = np.linspace(0,Nval-1,Nval)

#Loops over every N value up to a maximum, and calculates the L2 error, and plots the square of the error as a function
#of x.
for maxN in range(1, len(C_n)+1):
    error = quad(L2ErrorFunction, -1.0, 1.0, args=(maxN,))
    errorList.append(np.sqrt(error[0]))

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

#Error and Series Solution
error = quad(L2ErrorFunction, -1.0, 1.0, args=(len(C_n),))
error = np.sqrt(error[0])
seriesResult = LegendreSeries(xvals, len(C_n))

#Print Results
print "Legendre Series Coefficients:",  C_n
print "Error:", error

#Plot Results
plt.figure()
plt.plot(xvals, seriesResult, 'r', label='Legendre Series')
plt.plot(xvals, DesiredFuncton(xvals), 'b--', label='Desired Function')
plt.grid()
plt.legend()
plt.xlabel('X-Values')
plt.ylabel('Y-Values')
plt.title('Representing Functions Using Legendre Polynomials')
plt.show()


