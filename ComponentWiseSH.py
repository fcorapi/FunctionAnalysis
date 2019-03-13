#**************Representing Vectors Component-Wise Using Scalar SH Series*********************************

#Import Directories
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre, sph_harm, roots_legendre
from scipy.integrate import quad
from numpy.polynomial.legendre import legroots
from pylab import imshow
from mpl_toolkits import mplot3d
import time


#******************General Functions*****************************************

#Define conformal factor
def psi4(z,phi):
    factor = 1# - z**2
    return factor

#Define Legendre Function
def Legendre(x,n):
    leg = legendre(n)
    P_n = leg(x)
    return P_n

#Gauss-Legendre Quadrature Integration in 2-D
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

#*********Projection and Component Functions**********************************
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
    newMetricComp = 0
    row = 0
    col = 0
    if comp == 00:
        row = 0
        col = 0
    elif comp == 01:
        row = 0
        col = 1
    elif comp == 02:
        row = 0
        col = 2
    elif comp == 10:
        row = 1
        col = 0
    elif comp == 11:
        row = 1
        col = 1
    elif comp == 12:
        row = 1
        col = 2
    elif comp == 20:
        row = 2
        col = 0
    elif comp == 21:
        row = 2
        col = 1
    elif comp == 22:
        row = 2
        col = 2

    for loop3 in range(0, np.shape(metric(theta, phi))[0]):
        for loop4 in range(0, np.shape(metric(theta, phi))[1]):
            newMetricComp = (newMetricComp +
                             projOperator(theta, phi)[row,loop3] *
                             projOperator(theta, phi)[col,loop4] *
                             metric(theta, phi)[loop3,loop4])
    return newMetricComp

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

t = time.time()
print 'Finding coefficients...'

C_nXX = findCartCoeff(Nval, metricProj, intN, metricField, 00)
print 'XX Coefficients Found!'
C_nXY = findCartCoeff(Nval, metricProj, intN, metricField, 01)
print 'XY Coefficients Found!'
C_nXZ = findCartCoeff(Nval, metricProj, intN, metricField, 02)
print 'XZ Coefficients Found!'
#C_nYX = findCartCoeff(Nval, metricProj, intN, metricField, 10)
C_nYX = C_nXY
print 'YX Coefficients Found!'
C_nYY = findCartCoeff(Nval, metricProj, intN, metricField, 11)
print 'YY Coefficients Found!'
C_nYZ = findCartCoeff(Nval, metricProj, intN, metricField, 12)
print 'YZ Coefficients Found!'
#C_nZX = findCartCoeff(Nval, metricProj, intN, metricField, 20)
C_nZX = C_nXZ
print 'ZX Coefficients Found!'
#C_nZY = findCartCoeff(Nval, metricProj, intN, metricField, 21)
C_nZY = C_nYZ
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
#InvC_nYX = findCartCoeff(Nval, metricProj, intN, invMetricField, 10)
InvC_nYX = InvC_nXY
print 'YX Coefficients Found!'
InvC_nYY = findCartCoeff(Nval, metricProj, intN, invMetricField, 11)
print 'YY Coefficients Found!'
InvC_nYZ = findCartCoeff(Nval, metricProj, intN, invMetricField, 12)
print 'YZ Coefficients Found!'
#InvC_nZX = findCartCoeff(Nval, metricProj, intN, invMetricField, 20)
InvC_nZX = InvC_nXZ
print 'ZX Coefficients Found!'
#InvC_nZY = findCartCoeff(Nval, metricProj, intN, invMetricField, 21)
InvC_nZY = InvC_nYZ
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
#errorListyx = calcCartErrorList(C_nYX, Nval, metricProj, intN, metricField, 10)
errorListyx = errorListxy
print "Error for YX-Component Done!"
errorListyy = calcCartErrorList(C_nYY, Nval, metricProj, intN, metricField, 11)
print "Error for YY-Component Done!"
errorListyz = calcCartErrorList(C_nYZ, Nval, metricProj, intN, metricField, 12)
print "Error for YZ-Component Done!"
#errorListzx = calcCartErrorList(C_nZX, Nval, metricProj, intN, metricField, 20)
errorListzx = errorListxz
print "Error for ZX-Component Done!"
#errorListzy = calcCartErrorList(C_nZY, Nval, metricProj, intN, metricField, 21)
errorListzy = errorListyz
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
#invErrorListyx = calcCartErrorList(InvC_nYX, Nval, metricProj, intN, invMetricField, 10)
invErrorListyx = invErrorListxy
print "Error for YX-Component Done!"
invErrorListyy = calcCartErrorList(InvC_nYY, Nval, metricProj, intN, invMetricField, 11)
print "Error for YY-Component Done!"
invErrorListyz = calcCartErrorList(InvC_nYZ, Nval, metricProj, intN, invMetricField, 12)
print "Error for YZ-Component Done!"
#invErrorListzx = calcCartErrorList(InvC_nZX, Nval, metricProj, intN, invMetricField, 20)
invErrorListzx = invErrorListxz
print "Error for ZX-Component Done!"
#invErrorListzy = calcCartErrorList(InvC_nZY, Nval, metricProj, intN, invMetricField, 21)
invErrorListzy = invErrorListyz
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
seriesResultxx = SHSeries(np.cos(np.pi/2), np.pi/4, Nval, C_nXX)
seriesResultxy = SHSeries(np.cos(np.pi/2), np.pi/4, Nval, C_nXY)
seriesResultxz = SHSeries(np.cos(np.pi/2), np.pi/4, Nval, C_nXZ)
seriesResultyx = SHSeries(np.cos(np.pi/2), np.pi/4, Nval, C_nYX)
seriesResultyy = SHSeries(np.cos(np.pi/2), np.pi/4, Nval, C_nYY)
seriesResultyz = SHSeries(np.cos(np.pi/2), np.pi/4, Nval, C_nYZ)
seriesResultzx = SHSeries(np.cos(np.pi/2), np.pi/4, Nval, C_nZX)
seriesResultzy = SHSeries(np.cos(np.pi/2), np.pi/4, Nval, C_nZY)
seriesResultzz = SHSeries(np.cos(np.pi/2), np.pi/4, Nval, C_nZZ)
#Inverse Metric
invSeriesResultxx = SHSeries(np.cos(np.pi/2), np.pi/4, Nval, InvC_nXX)
invSeriesResultxy = SHSeries(np.cos(np.pi/2), np.pi/4, Nval, InvC_nXY)
invSeriesResultxz = SHSeries(np.cos(np.pi/2), np.pi/4, Nval, InvC_nXZ)
invSeriesResultyx = SHSeries(np.cos(np.pi/2), np.pi/4, Nval, InvC_nYX)
invSeriesResultyy = SHSeries(np.cos(np.pi/2), np.pi/4, Nval, InvC_nYY)
invSeriesResultyz = SHSeries(np.cos(np.pi/2), np.pi/4, Nval, InvC_nYZ)
invSeriesResultzx = SHSeries(np.cos(np.pi/2), np.pi/4, Nval, InvC_nZX)
invSeriesResultzy = SHSeries(np.cos(np.pi/2), np.pi/4, Nval, InvC_nZY)
invSeriesResultzz = SHSeries(np.cos(np.pi/2), np.pi/4, Nval, InvC_nZZ)

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

print 'Projection Operator: ', projOperator(np.pi/2,np.pi/4)
print 'Metric Projection XX: ', metricProj(np.pi/2,np.pi/4,identity,00)
print 'Metric Projection XY: ', metricProj(np.pi/2,np.pi/4,identity,01)
print 'Metric Projection XZ: ', metricProj(np.pi/2,np.pi/4,identity,02)
print 'Metric Projection YX: ', metricProj(np.pi/2,np.pi/4,identity,10)
print 'Metric Projection YY: ', metricProj(np.pi/2,np.pi/4,identity,11)
print 'Metric Projection YZ: ', metricProj(np.pi/2,np.pi/4,identity,12)
print 'Metric Projection ZX: ', metricProj(np.pi/2,np.pi/4,identity,20)
print 'Metric Projection ZY: ', metricProj(np.pi/2,np.pi/4,identity,21)
print 'Metric Projection ZZ: ', metricProj(np.pi/2,np.pi/4,identity,22)
#print 'SeriesResult*SeriesResult:', np.matmul(seriesResult,seriesResult)
print 'Metric Result', seriesResult
print 'Inverse Metric', invSeriesResult
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
# t = time.time()
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
#
# elapsedTime = time.time() - t
# print "Elapsed Time (s):", elapsedTime
