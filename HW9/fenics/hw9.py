# coding=utf-8
"""
FEniCS program: Solution of the heat equation with homogeneous Neumann boundary
conditions.
"""

from __future__ import print_function
from fenics import *
from mshr import *
import time

# Create a geometry and mesh it
square = Rectangle(Point(0., 0.), Point(1., 1.))
diskM = Circle(Point(0.5, 0.5), 0.2)
diskSW = Circle(Point(0.25, 0.), 0.2)
diskSE = Circle(Point(0.75, 0.), 0.2)
diskE = Circle(Point(1, 0.5), 0.2)
diskNE = Circle(Point(0.75, 1), 0.2)
diskNW = Circle(Point(0.25, 1), 0.2)
diskW = Circle(Point(0., 0.5), 0.2)
domain = square - diskM - diskSW - diskSE - diskE - diskNE - diskNW - diskW
mesh = generate_mesh(domain, 50) # h ~ 1/50

# Function space of linear finite elements
V = FunctionSpace(mesh, 'P', 1)

# Problem data
t0 = 0.0 # initial time (arbitrary)
# T = t0 + 5.0 # final time
T = t0 + 0.1
t = t0 # current time
a = 0.1 # thermal conductivity
u0 = interpolate(Constant(20.0), V) # initial temperature

# Refer to Theorem 2.3.25 to decide upon a sensible degree for the interpolation
# of the source term f below
f = Expression("t > tstop ? 0 : 200*exp(-5*x[0]*x[0]-2*x[1]*x[1])",
                degree = 1, t = t, tstop = t0 + 1.0)
#   From Theorem 2.3.25 we have that the error in the B norm, ||e||_B, is O(h)
#   for conforming linear finite elements if the order of the quadrature formula
#   is at least r = 2k-1, where k=1 is the order of the finite elements.
#   So, since order 1 quadrature integrates linear functions exactly and C²
#   functions O(h²) in 2D, if we take an order 1 approximation of f, the error
#   in approximating f will be O(h²) which is the same order as the error from
#   integrating the quadratic f*v, which is less than the O(h) error

# Parameters of the time-stepping scheme
theta = 0.0 # parameter for theta-method
# dt = 1e-2 # time step size
# dt = 1e-4
dt = 1.25e-4
tsteps = int(round((T-t0)/dt)) # number of time steps (round to ensure integer)

# Define the variational problem
u = TrialFunction(V)
v = TestFunction(V)
B = u*v*dx
if theta > 0.0:
    B += dt*theta*a*dot(grad(u), grad(v))*dx

# Export the initial data
u = Function(V, name='Temperature')
u.assign(u0)
# results = File('heat/thetaMethodBwdEuler.pvd')
# results = File('heat/thetaMethodFwdEuler.pvd')
# results = File('heat/thetaMethodCrankNicolson.pvd')
# results = File('heat/thetaMethodFwdEuler__T_0p1__dt_0p0001.pvd')
results = File('heat/thetaMethodFwdEuler__T_0p1__dt_0p000125.pvd')
results << (u, t)

# Time stepping
f.t = t0
for k in range(tsteps):

    t_last = t0 + k*dt # Last time
    t_curr = t0 + (k+1)*dt # Current time
    print('Step = ', k+1, '/', tsteps , 'Time =', t_curr)

    # Assemble the right hand side
    #   -> Currently, f.t == t_last from the last loop itereration
    L = (1-(1-theta)*dt*a)*u0*v*dx
    if theta < 1.0:
        L += dt*(1-theta)*f*v*dx

    # Update f and add contribution to L
    f.t = t_curr
    if theta > 0.0:
        L += dt*theta*f*v*dx

    # Compute the solution
    solve(B == L, u)
    results << (u, t_curr)

    # Update
    u0.assign(u)

# ---------------------------------------------------------------------------- #
# For comparing with BE for theta = 1.0
# ---------------------------------------------------------------------------- #

# # Export the initial data
# u0 = interpolate(Constant(20.0), V) # initial temperature
# U_BE = Function(V, name='Temperature')
# U_BE.assign(u0)
# results = File('heat/backwardEuler.pvd')
# results << (U_BE, t)
#
# # Time stepping
# for k in range(tsteps):
#
#     # Current time
#     t = t0 + (k+1)*dt
#     print('Step = ', k+1, '/', tsteps , 'Time =', t)
#
#     # Assemble the right hand side
#     f.t = t
#     L = (u0 + dt*f)*v*dx
#
#     # Compute the solution
#     solve(B == L, U_BE)
#     results << (U_BE, t)
#
#     # Update
#     u0.assign(U_BE)
#
# print('\n', '||u-U_BE||_L2 = ', errornorm(u, U_BE, 'L2'), '\n')
