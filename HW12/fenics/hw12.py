# coding=utf-8
"""
FEniCS program: Solution of the unsteady advection equation or advection-
diffusion equation with DG-discretisation in space.

Set D = 0.0 for advection problem with boundary data g
Set D > 0.0 for diffusion problem with no-flux Robin BC's (ignores g)
"""

from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
import time

# Create a domain and mesh
domain = Circle(Point(0., 0.), 1.) - Circle(Point(0., 1.15), 0.3) - \
         Circle(Point(0., -1.15), 0.3) + Circle(Point(0.85, 0.), 0.3) + \
         Circle(Point(-0.85, 0.), 0.3)
mesh = generate_mesh(domain, 100)
n = FacetNormal(mesh)

# Function space
r = 2
V = FunctionSpace(mesh, 'DG', r)

# Problem data
a = Expression(("-x[1]", "x[0]"), degree=1) # advection velocity
g = Constant(0.0) # boundary data
D = 0.01 # diffusion constant (set D = 0.0 for advection problem)
sigma = 0.1 # interior penalty parameter

u0_expr = Expression('pow(x[0],2)+pow(x[1]-0.5,2) < 1./16. ? 1. : 0.', degree=0)
u0 = project(u0_expr, V) # initial concentration

# Parameters of the time-stepping scheme
t0 = 0.0 # initial time
T = 2.0*pi # final time
t = t0 # current time
tsteps = 500 # number of time steps
dt = T/tsteps # time step size

# The advection operator has imaginary eigenvalues, suggesting that the
# Crank-Nicolson scheme is a good method and one should set θ = 0.5.
# It should be noted, however, that upwind discretisation introduces some
# numerical diffusion which ideally should be mitigated by using an anti-
# dissipative time stepping scheme (Forward Euler, etc.). I will just make the
# safe choice of θ = 0.5, which will be stable, and not try to guess the optimal
# amount of explicitness to add to the problem to counteract the diffusion.
#
#   Note: For the advection-diffusion problem, a fully implicit backward Euler
#         seems to fair much better, avoiding numerical dispersion arising from
#         the discontinuous initial data
theta = 1.0

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

def get_M(u,v):
    # regular M(u,v) mass matrix
    M = u*v*dx
    return M

def get_B(u,v):
    # regular B(u,v) term
    B = -u*dot(a,grad(v))*dx

    f = dot(a, n('+')) # take + normal direction
    f_pos = 0.5*(f + abs(f))
    f_neg = 0.5*(f - abs(f))

    B += jump(v)*u('+')*f_pos*dS + jump(v)*u('-')*f_neg*dS # interior jumps

    f = dot(a, n) # no ambiguity of n direction on boundary
    f_pos = 0.5*(f + abs(f))
    B += v*u*f_pos*ds # exterior edge contributions

    if abs(D) > 5*DOLFIN_EPS:
        # Add Diffusion term with no-flux Robin boundary condition
        h = CellDiameter(mesh)
        he = avg(h)

        B += D*dot(grad(u),grad(v))*dx # bilinear form of conforming methods
        B -= v*dot(u*a,n)*ds # consistency, boundary edges with: D∇u⋅n = (ua)⋅n
        B -= jump(v)*dot(avg(D*grad(u)),n('+'))*dS # consistency, int. edges
        B -= jump(u)*dot(avg(D*grad(v)),n('+'))*dS # symmetry
        B += (sigma/he)*jump(u)*jump(v)*dS # interior penalty

    return B

def get_Lg(u,v):
    if abs(D) > 5*DOLFIN_EPS:
        # Adv-diff problem: no boundary data (Robin no-flux condition is used)
        Lg = Constant(0.0)*v*dx
    else:
        # Advection problem: compute RHS source term
        f = dot(a, n) # no ambiguity of n direction on boundary
        f_neg = 0.5*(f - abs(f))
        Lg = -v*g*f_neg*ds

    return Lg

# Now, we have a system (thinking of M, B as matrices, and L,u as vectors):
#   M*du/dt + B*u = L
# (Note that L may be zero, as in the case of g=0 for the advection problem or
# for no-flux Robin BC's in the advection-diffusion problem)
#
# Writing f = L-B*u, the theta method gives us:
#   M*u1 = M*u0 + Δt(θ*f(t,u1) + (1-θ)*f(t,u0))
# And so:
#   M*u1 - θΔt*(L-B*u1) = M*u0 + (1-θ)Δt*(L-B*u0)
#    ==> (M + θΔt*B)*u1 = (M - (1-θ)Δt*B)*u0 + Δt*L

# LHS matrix: M + θΔt*B
M = get_M(u,v)
B = get_B(u,v)
A = assemble(M + theta*dt*B)

# I know that this source term is zero, but I thought I'd leave it in for
# generality, considering that the performance penalty for the vector addition
# is negligeable compared to the linear system solve anyways
Lg = get_Lg(u,v)
Lg_dt = assemble(dt*Lg)

# Create solver object for linear systems
solver = LUSolver(A) # Direct solver
solver.parameters.symmetric = False
solver.parameters.reuse_factorization = True

# Write initial data to file
u = Function(V, name='Concentration')
u.assign(u0)
concentration = File('hw12/advection.pvd')
concentration << (u, t)

for k in range(tsteps):
    # Current time
    t = t0 + (k+1)*dt
    print('Step = ', k+1, '/', tsteps , 'Time =', t)

    # Define right hand side: RHS = (M - (1-θ)Δt*B)*u0 + Δt*Lg
    M_u0 = get_M(u0,v)
    B_u0 = get_B(u0,v)
    L = assemble(M_u0 - (1-theta)*dt*B_u0) + Lg_dt
    solver.solve(u.vector(), L)

    # print(sum(assemble(get_M(u,v)).get_local())) # check mass conservation

    # Write data to file
    concentration << (u, t)

    # Update
    u0.assign(u)
