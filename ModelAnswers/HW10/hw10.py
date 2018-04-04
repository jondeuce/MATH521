# coding=utf-8
"""
FEniCS program: Solution of the wave equation with homogeneous Dirichlet (reflective)
boundary conditions.
"""

from __future__ import print_function
from fenics import *
from mshr import *
import time
import csv

# Create a mesh on the unit disk
disk = Circle(Point(0., 0.), 1.)
mesh = generate_mesh(disk, 100) # h ~ 1/50

# Function space and boundary condition
V = FunctionSpace(mesh, 'P', 1)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(0.), boundary)

# Problem data
t0 = 0. # initial time
T = 5. # final time
t = t0 # current time
c = Constant(1.) # propagation speed
u0 = interpolate(Expression('pow(x[0],2)+pow(x[1],2) < 1./16. ? pow(1.-16.*(pow(x[0],2)+pow(x[1],2)),2) : 0.', degree=1), V) # initial displacement
v0 = interpolate(Constant(0.), V) # initial velocity

# Parameters of the time-stepping scheme
tsteps = 500 # number of time steps
dt = T/tsteps # time step size
theta = Constant(0.5) # degree of implicitness

# Define the variational problem
w = TrialFunction(V) # w = u in the 1st equation and w = v in the 2nd equation
z = TestFunction(V)
B1 = (w*z + (theta*c*dt)**2*dot(grad(w), grad(z)))*dx # LHS of the 1st equation
B2 = w*z*dx # LHS of the 2nd equation

# Set initial data
u = Function(V, name='Displacement')
v = Function(V, name='Velocity')
u.assign(u0)
v.assign(v0)
T = assemble(.5*v**2*dx)
V = assemble(.5*c**2*dot(grad(u), grad(u))*dx)
E = T + V

# Write initial data to file
displacement = File('wave/theta.pvd')
displacement << (u, t)
energy = csv.writer(open('wave/energy.csv', 'w'))
energy.writerow([t] + [E] + [T] + [V])
# Import in Octave / MATLAB with command
#   A = csvread('wave/energy.csv');
# Or, if your FEniCS installation has graphics output enabled, you could also plot directly from this file using Python commands

# Time stepping
for k in range(tsteps):

    # Current time
    t = t0 + (k+1)*dt
    print('Step = ', k+1, '/', tsteps , 'Time =', t)

    # System for the displacement 
    L1 = ((u0 + dt*v0)*z - (theta*(1.-theta)*(dt*c)**2)*dot(grad(u0), grad(z)))*dx
    solve(B1 == L1, u, bc)
    
    # System for the velocity
    L2 = (v0*z - dt*c**2*dot(grad(theta*u+(1.-theta)*u0), grad(z)))*dx
    solve(B2 == L2, v, bc)

    # Compute energy
    T = assemble(.5*v**2*dx)
    V = assemble(.5*c**2*dot(grad(u), grad(u))*dx)
    E = T + V

    # Write data to file
    displacement << (u, t)
    energy.writerow([t] + [E] + [T] + [V])

    # Update
    u0.assign(u)
    v0.assign(v)