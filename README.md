# FunctionAnalysis
Representing functions as Legendre Series, calculating derivatives using Legendre series, etc. Also representing scalar and vector fields
on a round 2-sphere using scalar and vector spherical harmonics, as well as calculating some properties of a black hole horizon
 (to be added).


The main program is entitled SphericalHarmonicsDE, and can be used to represent scalar fields on  a 2-sphere,
solving Poisson's equation by solving a system of linear equations, and represents vector fields on a 2-sphere.
Calculating different properties of the sphere (black hole horizon), such as angular momentum, mass and spin is also implemented,
however, the one form omega that is required for the angular momentum calculations must be found analytically and manually inputted into
the code. Also, this can only be done on a round 2-sphere, until conformally spherical metrics are implemented into the code.

Scalar Fields:
To represent scalar fields as a series of spherical harmonics, comment out lines 531 - 717 (sections entitled "Representing
Desired Vector Function" and "Solving Poisson's Equation") , and uncomment lines 450 -526 (Section "Representin Desired
Function"). The value for Nval on line 437 is equal to the l-value of the spherical harmonic series you want it to go up to. 
This can be modified depending on how long you want the series to be. To change which function you want to be represented, 
modify the function "DesiredFunction" on line 28, to a function of theta and phi on the sphere. 

Poisson's Equation:
To solve Poisson's equation, comment out lines 448-637 (sections entitled "Representing Desired Vector Function" and 
"Representing Desired Function") and uncomment lines 641- 717 (section entitled "Solving Poisson's Equation"). Similarly, the value
for Nval on line 437 will specificy which l-value the series will go up to. For Possion's equation, \nabla^2\phi = \rho, the rho function,
is to be known to the user, can be specified on line 49 (rho function), and for testing purposes, the analytic function for phi
can be modified on line 40 (phi function). The rho_0 variable (on line 644) is the constant term for the phi function desired (i.e the 
integration constant for the problem, and is to be specified by the user.

Vector Fields:
To represent vector fields as a series of vector spherical harmonics, comment out lines 450-526 and 641-717, and uncomment lines 531-637.
As of now, the function that you wish to represent must be known to be either polar or axial. The desired function can be modifed in the
function entitled "VecDesiredFunction" on line 252. If the field is polar define it in the polar condition, and if it is axial, define 
it in the axial condition. Then modify the vecKind variable on line 532 to either 'axial' or 'polar' depending on your function. (This 
will be changed in the future.)

Angular Momentum:
Still being worked on. 
