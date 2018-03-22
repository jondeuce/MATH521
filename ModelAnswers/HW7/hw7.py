"""
FEniCS program: steady reaction-advection-diffusion equation with Dirichlet
conditions.

The boundary values and the source term are chosen such that
    u(x,y) = 1 + x^2 + 2y^2
is the exact solution of this problem.

Adapted from ft01_posson.py available at
https://fenicsproject.org/pub/tutorial/html/._ftut1004.html
"""

from __future__ import print_function
from fenics import *

# Create mesh and define function space
mesh = UnitSquareMesh(8,8)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
# NB: interpolation of this quadratic function with piecewise quadratics is exact

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define parameters
D = Constant(10)
a = Constant((1,1))
r = Constant(1)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression('x[0]*x[0] + 2*x[1]*x[1] + 2*x[0] + 4*x[1] - 59', degree=2)
# NB: interpolation of this quadratic function with piecewise quadratics is exact
B = D*dot(grad(u), grad(v))*dx + dot(a, grad(u))*v*dx + r*u*v*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(B == L, u, bc)

# Save solution to file in VTK format
vtkfile = File('hw7.pvd')
vtkfile << u

# Compute error in H1 and L2 norm
error_H1 = errornorm(u_D, u, 'H1') # u_D is exact (see above) -> ignore warning
error_L2 = errornorm(u_D, u, 'L2') # u_D is exact (see above) -> ignore warning

# Print errors
print('error_H1 =', error_H1)
print('error_L2  =', error_L2)
