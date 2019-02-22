import numpy as np
import matplotlib.pyplot as plt

thetaVals = np.linspace(0, np.pi,20)
phiVals = np.linspace(0,2*np.pi,20)
oldVals = [thetaVals,phiVals]

thetaMesh, phiMesh = np.meshgrid(thetaVals, phiVals)
oldValsMesh = [thetaMesh,phiMesh]

#transform vector
vecTheta= 0*thetaMesh
vecPhi = np.power(np.sin(thetaMesh),2)
vecTransform = [vecTheta,vecPhi]

newVals = np.add(oldValsMesh,vecTransform)

newThetaMesh, newPhiMesh = np.meshgrid(newVals[0],newVals[1])

vecThetaMesh = 0*thetaMesh
vecPhiMesh = np.power(np.sin(thetaMesh),2)
vectorTransformMesh = [vecThetaMesh,vecPhiMesh]

#newValsMesh = oldValsMesh + vectorTransformMesh
#newThetaMesh = newValsMesh[0]
#newPhiMesh = newValsMesh[1]

plt.scatter(phiMesh,thetaMesh)
plt.xlabel('$\phi$-Values')
plt.ylabel('$\\theta$-Values')
plt.gca().invert_yaxis()

plt.figure()
plt.scatter(newVals[1],newVals[0])
plt.xlabel('$\Phi$-Values')
plt.ylabel('$\\Theta$-Values')
plt.ylim([0,np.pi])
plt.gca().invert_yaxis()

plt.figure()
plt.quiver(phiMesh[::4,::4], thetaMesh[::4,::4], vectorTransformMesh[1][::4,::4], -vectorTransformMesh[0][::4,::4], color = 'r')
plt.gca().invert_yaxis()
plt.show()