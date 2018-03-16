# coding=utf-8
"""
FEniCS program: Solution of the heat equation with homogeneous Neumann boundary
conditions.
"""

from __future__ import print_function
from fenics import *
from mshr import *
import time
import numpy as np

def create_heateqn_problem(t0 = 0.0, T = 5.0, dt = 1e-2):
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
    u0 = interpolate(Constant(20.0), V) # initial temperature
    T_source = 1.0 # source f is active for the first 'T_source' seconds

    # Parameters of the time-stepping scheme
    t = t0 # current time
    tsteps = int(round(T/dt)) # number of time steps (round to ensure integer)

    # Refer to Theorem 2.3.25 to decide upon a sensible degree for the interpolation
    # of the source term f below
    #   From Theorem 2.3.25 we have that the error in the B norm, ||e||_B, is O(h)
    #   for conforming linear finite elements if the order of the quadrature formula
    #   is at least r = 2k-1, where k=1 is the order of the finite elements.
    #   So, since order 1 quadrature integrates linear functions exactly and C²
    #   functions O(h²) in 2D, if we take an order 1 approximation of f, the error
    #   in approximating f will be O(h²) which is the same order as the error from
    #   integrating the quadratic f*v, which is less than the O(h) error
    f = Expression("t > tstop ? 0 : 200*exp(-5*x[0]*x[0]-2*x[1]*x[1])",
    degree = 1, t = t, tstop = t0 + T_source)

    return u0, f, V, t, tsteps


def heateqn_bwdeuler(t0 = 0.0, T = 5.0, dt = 1e-2, a = 0.1, prnt = True,
                     save = False, fname = 'heat/bwdeuler_theta.pvd'):
    # Create problem geometry
    u0, f, V, t, tsteps = create_heateqn_problem(t0 = 0.0, T = 5.0, dt = 1e-2)

    # Define the variational problem for the Backward Euler scheme
    u = TrialFunction(V)
    v = TestFunction(V)
    B = u*v*dx + dt*a*dot(grad(u), grad(v))*dx

    # Export the initial data
    u = Function(V, name='Temperature')
    u.assign(u0)
    if save:
        results = File('heat/backwardEuler.pvd')
        results << (u, t)

    # Time stepping
    for k in range(tsteps):

        # Current time
        t = t0 + (k+1)*dt
        if prnt:
            print('Step = ', k+1, '/', tsteps , 'Time =', t)

        # Assemble the right hand side
        f.t = t
        L = (u0 + dt*f)*v*dx

        # Compute the solution
        solve(B == L, u)
        if save:
            results << (u, t)

        # Update
        u0.assign(u)

    # Return Solution
    return u

def heateqn_thetamethod(t0 = 0.0, T = 5.0, dt = 1e-2, a = 0.1, theta = 1.0,
                            save = False, fname = 'heat/bwdeuler_theta.pvd',
                            prnt = True, cmp = False):
    # Create problem geometry
    u0, f, V, t, tsteps = create_heateqn_problem(t0 = 0.0, T = 5.0, dt = 1e-2)

    # Define the variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    B = u*v*dx
    if theta > 0.0:
        B += dt*theta*a*dot(grad(u), grad(v))*dx

    # Export the initial data
    u = Function(V, name='Temperature')
    u.assign(u0)
    if save:
        results = File(fname)
        results << (u, t)

    # Time stepping
    f.t = t0
    for k in range(tsteps):

        t_last = t0 + k*dt # Last time
        t_curr = t0 + (k+1)*dt # Current time
        if prnt:
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
        if save:
            results << (u, t_curr)

        # Update
        u0.assign(u)

    if cmp:
        # Compare the norm of the difference between the solution calculated
        # with the theta-method and with the backward euler method
        u_BE = heateqn_bwdeuler(t0 = t0, T = T, dt = dt, a = a,
                                prnt = prnt, save = save)
        print('\n')
        print('u_th_max = ', np.max(np.abs(u.vector().get_local())))
        print('u_BE_max = ', np.max(np.abs(u_BE.vector().get_local())))
        print('||u-u_BE||_L2 = ', errornorm(u, u_BE, 'L2'))
        print('\n')

# ---------------------------------------------------------------------------- #
# Stricter log level to avoid so much printing
# ---------------------------------------------------------------------------- #
set_log_level(30)

# ---------------------------------------------------------------------------- #
# Q2(b): theta-method - backward euler consistency check with Q2(a)
# ---------------------------------------------------------------------------- #
heateqn_thetamethod(t0 = 0.0, T = 5.0, dt = 1e-2, a = 0.1, theta = 1.0,
                    fname = 'heat/thetaMethodBwdEuler.pvd',
                    save = True, cmp = True, prnt = True)

# ---------------------------------------------------------------------------- #
# Q2(b): theta-method - forward euler and Crank-Nicolson
# ---------------------------------------------------------------------------- #
heateqn_thetamethod(t0 = 0.0, T = 5.0, dt = 1e-2, a = 0.1, theta = 0.0,
                    fname = 'heat/thetaMethodFwdEuler.pvd',
                    save = True, cmp = True, prnt = True)

heateqn_thetamethod(t0 = 0.0, T = 5.0, dt = 1e-2, a = 0.1, theta = 0.5,
                    fname = 'heat/thetaMethodCrankNicolson.pvd',
                    save = True, cmp = True, prnt = True)

# ---------------------------------------------------------------------------- #
# Q2(c): theta-method - forward euler, shorter duration T and timestep dt
# ---------------------------------------------------------------------------- #
heateqn_thetamethod(t0 = 0.0, T = 0.1, dt = 1.25e-4, a = 0.1, theta = 0.0,
                    fname = 'heat/thetaMethodFwdEuler__T_0p1__dt_0p000125.pvd',
                    save = True, cmp = True, prnt = True)

heateqn_thetamethod(t0 = 0.0, T = 0.1, dt = 1.00e-4, a = 0.1, theta = 0.0,
                    fname = 'heat/thetaMethodFwdEuler__T_0p1__dt_0p0001.pvd',
                    save = True, cmp = True, prnt = True)
