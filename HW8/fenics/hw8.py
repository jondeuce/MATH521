# coding=utf-8
"""
FEniCS program: A posteriori error estimates for Poisson's equation with
Dirichlet boundary conditions und the unit square.

The boundary values and the source term are chosen such that

    u(x,y) = x(1-xᵃ)y(1-y)

is the exact solution of this problem (with a parameter a >=1).
In particular, for the region of interest R = (1/2, 1) x (0,1/2), we have

    f(x,y) = a(a+1)xᵃ⁻¹y(1-y) + 2x(1-xᵃ)
      J(u) = (3a-2+2¹⁻ᵃ)/(24a + 48)

where J(u) is the average of the solution u over R.
"""

from __future__ import print_function
from fenics import *
import numpy as np

###############################################################################
# DATA
###############################################################################

# Parameters
N = 64 # the PDE will be solved on an NxN grid (N must be even)
a = 100.
f = Expression('a*(a+1)*pow(x[0],a-1)*x[1]*(1-x[1]) + 2*x[0]*(1-pow(x[0],a))',
               degree=3, a=a) # source term
# NB: For solving the PDE with the optimal convergence rate, a piecewise
# constant approximation (midpoint rule) would already suffice. However, we'll
# also need f later on to compute the (ideally exact) residuals, and hence
# we're using a higher-order, piecewise cubic, interpolation here.
u_D = Constant(0.0) # boundary values

# Exact solution
uBar = Expression('x[0]*(1-pow(x[0],a))*x[1]*(1-x[1])', degree=3, a=a)
# NB: by interpolating the exact solution with piecewise polynomials of degree
#    (degree of FE solution) + 2 = 3
# we hope that the interpolation error between the exact solution and the
# interpolated uBar will be negligible compared to the error between the exact
# solution and the piecewise linear FE solution.

# Exact quantity of interest
JBar = (3*a - 2 + 2.0**(1-a))/(24*a + 48)

# Create mesh and compute extra mesh data
coarsemesh = UnitSquareMesh(N/2,N/2) # post-processing only (cheap Strategy 2)
mesh = refine(coarsemesh) # for solving PDEs (each triangle -> 4 subtriangles)
h = CellDiameter(mesh) # longest edge of each triangle
n = FacetNormal(mesh) # outward pointing normal vectors on each triangle edge

###############################################################################
# SOLUTION OF PROBLEM (P)
###############################################################################

# Function space and boundary conditions
V = FunctionSpace(mesh, 'P', 1)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
B = dot(grad(u), grad(v))*dx
F = f*v*dx

# Compute solution
u = Function(V, name='primal solution')
solve(B == F, u, bc)

# Compute quantity of interest
#   Hint: go back to the tutorial
#   https://fenicsproject.org/pub/tutorial/html/._ftut1004.html#___sec29
#   and refer to the green box "String expressions must have valid C++ syntax"
#   to find out how you can input a function that's defined piecewise.
x0, x1 = MeshCoordinates(mesh)
Inv_R  = 1.0/0.25

# Since the domain is ]0,1[ x ]0,1[, we may assume all x0, x1 values are in
# this range, and so we can specify the lower left corner R = ]1/2,1[ x ]0,1/2[
# simply as 0.5 <= x0 and x1 <= 0.5. Below, the multiplication of the two
# conditional expressions results in an expression which is 1 in R, and zero
# outside of R.
j = Inv_R * conditional(0.5 <= x0, 1, 0) * conditional(x1 <= 0.5, 1, 0)
Jh = assemble(j*u*dx) # quantity of interest computed from numerical solution

# L²-error
error_L2 = errornorm(uBar, u, 'L2')
# J-error
error_J = np.abs(Jh-JBar)

###############################################################################
# A POSTERIORI ERROR ESTIMATION: L²-NORM
###############################################################################

# All error indicators and cell-wise norms are constant on each triangle
W = FunctionSpace(mesh, 'DG', 0) # space of piecewise constant functions
w = TestFunction(W) # hat functions which 1 on one triangle and = 0 elsewhere

# (Squares of) local error indicators
etaT_L2 = ( h**4 * (-div(grad(u))-f)**2 * w * dx +
             0.5 * avg(h)**3 * jump(grad(u),n)**2 * avg(w) * dS )
# NB: For the test function w which is = 1 on triangle T, this expression
# returns the error indicator on this triangle T.
#    avg(w) = average of
#                (w evaluated on one side of the edge)
#             and
#                (w evaluated on the other side of the edge)
#
#             | 1/2 (on the edges of triangle T)
#           = |
#             | 0 (on all other edges)

# Assemble the vector which contains the (squares of) local error indicators
etaT_L2_vec = np.abs(assemble(etaT_L2))
# NB: etaT_L2 is a linear form like <f,v>
#     etaT_L2_vec is a vector like the load vector fh
# Here the absolute value only makes sure that none of the error indicators are
# negative (which could happen due to round-off errors).

# A posteriori error estimate of the L²-error
eta_L2 = np.sqrt(np.sum(etaT_L2_vec))

print('=== A POSTERIORI ERROR ESTIMATION: L²-NORM ===========================')
print('\n')
print('L²-error         ||eh|| =', error_L2)
print('estimator          η_L² =', eta_L2)
print('ratio       η_L²/||eh|| =', eta_L2/error_L2)
print('\n')


###############################################################################
# A POSTERIORI ERROR ESTIMATION: QUANTITY OF INTEREST
###############################################################################

## EXPENSIVE STRATEGY 1 ########################################################

# Function space and boundary conditions
Z_fine = FunctionSpace(mesh, 'P', 2)
bc = DirichletBC(Z_fine, u_D, boundary)

# Define variational problem
zh = TrialFunction(Z_fine)
v = TestFunction(Z_fine)
B = dot(grad(zh), grad(v))*dx
J = j*v*dx
# NB: We have to re-assemble the entire system from scratch since we're now
# using quadratic and not linear elements. Cannot re-use anything from the
# primal problem. It even has more degrees of freedom -> bigger linear system.

# Compute solution
zh = Function(Z_fine, name='dual solution')
solve(B == J, zh, bc) # numerical solution of the dual problem

# Piecewise quadratic approximation of the exact solution
z = zh
# Its piecewise linear interpolant
Iz = interpolate(z,V)

# NB: In FEniCS we can only add / subtract functions that belong to the same
# function space. We need z-Iz, however:
#    z belongs to Z_fine = {piecewise quadratic functions on mesh}
#    Iz belongs to V = {piecewise linear functions on mesh}
# Therefore, we re-write the piecewise linear function as a piecewise quadratic
# function (with quadratic terms = 0):
Iz = interpolate(Iz,Z_fine)

# Local error indicators
etaT_J1 = (-div(grad(u))-f)*(z-Iz)*w*dx + jump(grad(u),n)*(z-Iz)*avg(w)*dS

# Assemble the vector which contains the local error indicators
etaT_J1_vec = np.abs(assemble(etaT_J1))

# Expensive a posteriori error estimate of the J-error
eta_J1 = np.sum(etaT_J1_vec)

print('=== A POSTERIORI ERROR ESTIMATION: QUANTITY OF INTEREST (EXPENSIVE) ==')
print('\n')
print('J-error         |J(eh)| =', error_J)
print('estimator          η_J1 =', eta_J1)
print('ratio      η_J1/|J(eh)| =', eta_J1/error_J)
print('\n')


## CHEAP STRATEGY 2 ############################################################

# Function space and boundary conditions
Z_fine   = FunctionSpace(mesh, 'P', 2)
Z_coarse = FunctionSpace(coarsemesh, 'P', 2) # quadratic space on courser mesh
bc = DirichletBC(V, u_D, boundary)

# Define variational problem
zh = TrialFunction(V)
v = TestFunction(V)
B = dot(grad(zh), grad(v))*dx
J = j*v*dx
# NB: If we hadn't overwritten the data from the primal problem for the
# expensive error estimator, we could have re-used it here. FEniCS even
# provides the command 'adjoint' which computes the left hand side of the dual
# problem automatically from the bilinear form B of the primal problem.

# Compute solution
zh = Function(V, name='dual solution')
solve(B == J, zh, bc) # numerical solution of the dual problem

# Piecewise quadratic interpolant on patches of four triangles
z = interpolate(zh, Z_coarse) # Its piecewise quadratic interpolant on Z_coarse

# Its piecewise linear interpolant = ???
Iz = zh

# How do you compute their difference???
# NB: In FEniCS we can only add / subtract functions that belong to the same
# function space. We need z-Iz, however:
#    z belongs to Z_coarse = {piecewise quadratic functions on coarsemesh}
#    Iz belongs to V = {piecewise linear functions on mesh}
# Therefore, we interpolate the piecewise quadratic function z on the coarsemesh
# AND the piecewise linear function Iz on the fine mesh to the piecewise
# quadratic space Z_fine on the fine mesh; this doesn't change either of the
# function's actual shape, only puts them into the same space

# Testing that the functions don't change during interpolation:
# print('L2-error: ||z  - interpolate(z, Z_fine)|| = ', assemble( (interpolate(z, Z_fine) - z )**2 * dx(mesh) ) )
# print('L2-error: ||Iz - interpolate(Iz,Z_fine)|| = ', assemble( (interpolate(Iz,Z_fine) - Iz)**2 * dx(mesh) ) )

Iz = interpolate(Iz,Z_fine) # interp fine piecewise linear to piecewise quad
z  = interpolate(z,Z_fine)  # interp coarse piecewise quad to fine piecewise quad

# Local error indicators
etaT_J2 = (-div(grad(u))-f)*(z-Iz)*w*dx + jump(grad(u),n)*(z-Iz)*avg(w)*dS

# Assemble the vector which contains the local error indicators
etaT_J2_vec = np.abs(assemble(etaT_J2))

# Cheap a posteriori error estimate of the J-error
eta_J2 = np.sum(etaT_J2_vec)

print('=== A POSTERIORI ERROR ESTIMATION: QUANTITY OF INTEREST (CHEAP) ======')
print('\n')
print('J-error         |J(eh)| =', error_J)
print('estimator          η_J2 =', eta_J2)
print('ratio      η_J2/|J(eh)| =', eta_J2/error_J)
print('\n')


###############################################################################
# EXPORT DATA FOR PLOTTING
###############################################################################

# Exporting data from FEniCS to post-process it in ParaView works as follows:
#    1. Start with a function u.
#    2. Use the left shift command << to export it to a PVD file.
#          Piecewise constant functions are saved as values on triangles.
#          Piecewise linear functions are saved as values on the mesh points.
#          Piecewise quadratic and higher-order functions are also saved as
#          values on the mesh points only, so they will be plotted as piecewise
#          linear NOT piecewise quadratic etc.
#    3. Open the PVD file in ParaView. You normally have to click on the green
#       button 'Apply' before the function is shown.
#    4. For a 3D surface plot, go to Filters -> Alphabetical -> Warp by Scalar
#       and click 'Apply'. In the figure window, switch from '2D' to '3D' so
#       that you can rotate the surface.

# Forcing function
File('poisson/hw8_f.pvd') << interpolate(f, Z_fine)

# Solution of the primal problem
File('poisson/hw8_u.pvd') << u

# Solution of the dual problem
File('poisson/hw8_z.pvd') << zh

# Cell residuals
# We have to compute a piecewise constant function with the cell residuals as
# its function values.

# Squares of cell residuals ||rT||²_L²(T)
rT = (-div(grad(u))-f)**2 * w * dx
# Assemble the corresponding vector
rT_vec = np.sqrt(np.abs(assemble(rT)))
# Define the function with these function values
rT_fun = Function(W, name='cell residuals')
rT_fun.vector().set_local(rT_vec)
File('poisson/hw8_rT.pvd') << rT_fun

# Squares of dual weights ||wT||²_L²(T)
wT = (z-Iz)**2 * w * dx
# Assemble the corresponding vector
wT_vec = np.sqrt(np.abs(assemble(wT)))
# Define the function with these function values
wT_fun = Function(W, name='dual weights')
wT_fun.vector().set_local(wT_vec)
File('poisson/hw8_wT.pvd') << wT_fun

# Error indicators
etaT = (-div(grad(u))-f)*(z-Iz)*w*dx + jump(grad(u),n)*(z-Iz)*avg(w)*dS
# Assemble the corresponding vector
etaT_vec = np.sqrt(np.abs(assemble(etaT)))
# Define the function with these function values
etaT_fun = Function(W, name='error indicators')
etaT_fun.vector().set_local(etaT_vec)
File('poisson/hw8_etaT.pvd') << etaT_fun
