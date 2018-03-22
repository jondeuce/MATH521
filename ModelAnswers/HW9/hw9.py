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
mesh = generate_mesh(domain, 200) # h ~ 1/50

# Function space of linear finite elements
V = FunctionSpace(mesh, 'P', 1)

# Problem data
t0 = 0. # initial time
T = 5. # final time
t = t0 # current time
a = Constant(0.1) # thermal conductivity
u0 = interpolate(Constant(20.0), V) # initial temperature
f = Expression('t > 1. ? 0. : 200.*exp(-5.*pow(x[0],2)-2.*pow(x[1],2))', degree=1, t = t) # source term
# Refer to Theorem 2.3.25 to decide upon a sensible degree for the interpolation of this function.
# -> degree = 0 or 1

# Parameters of the time-stepping scheme
tsteps = 500 # number of time steps
dt = T/tsteps # time step size
theta = Constant(0.5) # degree of implicitness

# Define the variational problem
u = TrialFunction(V)
v = TestFunction(V)
B = (u*v + theta*dt*a*dot(grad(u), grad(v)))*dx

# Export the initial data
u = Function(V, name='Temperature')
u.assign(u0)
results = File('heat/theta.pvd')
results << (u, t)

# Time stepping
for k in range(tsteps):

    # Current time
    t = t0 + (k+1)*dt
    print('Step = ', k+1, '/', tsteps , 'Time =', t)
    
    # Assemble the right hand side
    L = u0*v*dx + (1.-theta)*dt*(f*v - a*dot(grad(u0), grad(v)))*dx
    f.t = t
    L += theta*dt*f*v*dx

    # Compute the solution
    solve(B == L, u)
    results << (u, t)

    # Update
    u0.assign(u)
