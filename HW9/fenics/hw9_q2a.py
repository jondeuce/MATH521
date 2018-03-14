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
T = t0 + 5.0 # final time
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
dt = 1e-2 # time step size
tsteps = int(round((T-t0)/dt)) # number of time steps (round to ensure integer)

# Define the variational problem
u = TrialFunction(V)
v = TestFunction(V)
B = u*v*dx + dt*a*dot(grad(u), grad(v))*dx

# Export the initial data
u = Function(V, name='Temperature')
u.assign(u0)
results = File('heat/backwardEuler.pvd')
results << (u, t)

# Time stepping
for k in range(tsteps):

    # Current time
    t = t0 + (k+1)*dt
    print('Step = ', k+1, '/', tsteps , 'Time =', t)

    # Assemble the right hand side
    f.t = t
    L = (u0 + dt*f)*v*dx

    # Compute the solution
    solve(B == L, u)
    results << (u, t)

    # Update
    u0.assign(u)
