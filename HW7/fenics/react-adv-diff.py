# This Python file uses the following encoding: utf-8
from __future__ import print_function
from fenics import *
import os, sys
import numpy as np

"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.

  -DΔu + ∇⋅(au) + ru = f    in the unit square
                   u = g    on the boundary

    g = x_1*(1-x_1) + x_2*(1-x_2)
    D = 1
    a = [1; 1]
    r = 1
    f = 6 - x_1*(x_1+1) - x_2*(x_2+1)
"""

# Create mesh and define function space
N = 8 # repeat for N = 4, 8, 16, 32
mesh = UnitSquareMesh(N, N)
V = FunctionSpace(mesh, 'P', 1) # degree 2 recovers exact solution, as expected

# Define boundary condition
g = Expression('x[0]*(1-x[0]) + x[1]*(1-x[1])', degree=2, domain=mesh)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, g, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
D = Constant(1.0)
a = Constant((1.0,1.0))
r = Constant(1.0)
f = Expression('6 - x[0]*(x[0]+1) - x[1]*(x[1]+1)', degree=2, domain=mesh)
B = D*dot(grad(u), grad(v))*dx + (div(a*u) + r*u)*v*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(B == L, u, bc)

# Compute maximum error at vertices
vertex_values_u_D = g.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Compute error in H1 and L2 norm
error_H1 = errornorm(g, u, 'H1')
error_L2 = errornorm(g, u, 'L2')

# Print errors
print('error_H1  =', error_H1)
print('error_L2  =', error_L2)
print('error_max =', error_max)

# Plot solution and mesh
plot(u, title='Finite Element Solution')
plot(mesh, title='Finite Element Mesh')

# Save solution to file in VTK format
vtkfile = File('react-adv-diff/solution.pvd')
vtkfile << u
