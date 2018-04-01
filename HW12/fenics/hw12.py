# coding=utf-8
"""
FEniCS program: Solution of the unsteady advection equation with a DG-discretisation in space.
"""

from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
import time

# Create a domain and mesh
domain = Circle(Point(0., 0.), 1.) - Circle(Point(0., 1.15), 0.3) - Circle(Point(0., -1.15), 0.3) + Circle(Point(0.85, 0.), 0.3) + Circle(Point(-0.85, 0.), 0.3) 
mesh = generate_mesh(domain, 100)
n = FacetNormal(mesh)

# Function space
V = FunctionSpace(mesh, 'DG', ???)

# Problem data
a = Expression(("-x[1]", "x[0]"), degree=1) # advection velocity
g = Constant(0.) # boundary data
u0 = project(Expression('pow(x[0],2)+pow(x[1]-0.5,2) < 1./16. ? 1. : 0.', degree=0), V) # initial concentration

# Parameters of the time-stepping scheme
t0 = 0. # initial time
T = 2.*pi # final time
t = t0 # current time
tsteps = 500 # number of time steps
dt = T/tsteps # time step size
theta = ???

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

???

# Create solver object for linear systems
solver = ???
solver.parameters.???

# Write initial data to file
u = Function(V, name='Concentration')
u.assign(u0)
concentration = File('hw12/advection.pvd')
concentration << (u, t)

for k in range(tsteps):

    # Current time
    t = t0 + (k+1)*dt
    print('Step = ', k+1, '/', tsteps , 'Time =', t)

    # Define right hand side
    ???
    
    solver.solve(???)

    # Write data to file
    concentration << (u, t)

    # Update
    u0.assign(u)
