# coding=utf-8
"""
FEniCS program: Solution of the Bloch-Torrey equation with Neumann boundary
conditions.
"""

from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
import time
import csv
import os
from timeit import default_timer as timer

def create_bt_problem(loadmesh = True, savemesh = True, N = 100, nslice = 32,
    L = 3000.0, vesselrad = 250.0, vesselunion = False):
    # voxel parameters (one major vessel)
    #    L [= 3000.0]        voxel width [um]
    #    vesselrad [= 250.0] major vessel radius [um]
    #    N [= 100]           approx. num. cells along box edge (mesh resolution)
    #    nslice [= 32]       num. edges used to construct cylinder boundary


    mesh_str = 'cylinder_N' + str(N) + '_ns' + str(nslice) + '_r' + str(int(round(vesselrad)))
    geom_foldname = 'bt/geom/union' if vesselunion else 'bt/geom/hollow';
    mesh_foldname = geom_foldname + '/' + mesh_str
    mesh_filename = mesh_foldname + '/' + mesh_str + '.xml.gz'

    if loadmesh and os.path.isfile(mesh_filename):
        mesh = Mesh(mesh_filename)
    else:
        # Create a geometry and mesh it
        Nx, Ny, Nz = N, N, N
        Lx, Ly, Lz = L, L, L

        # cylindrical vessel radius [um]
        voxel = Box(Point(-Lx/2, -Ly/2, -Lz/2), Point(Lx/2, Ly/2, Lz/2))
        vessel = Cylinder(Point(0,0,-Lz/2), Point(0,0,Lz/2), vesselrad, vesselrad, segments = nslice)

        if vesselunion:
            domain = voxel + vessel
        else:
            domain = voxel - vessel

        mesh_res = N
        mesh = generate_mesh(domain, mesh_res)

        if savemesh:
            if not os.path.exists(mesh_foldname):
                os.makedirs(mesh_foldname)
            File(mesh_filename) << mesh

    # Define function space
    # element = VectorElement('P', tetrahedron, 1, dim=2)
    # V = FunctionSpace(mesh, element)
    # P1 = FiniteElement('P', tetrahedron, 1)
    # element = MixedElement([P1, P1])
    # V = FunctionSpace(mesh, element)
    # V = VectorFunctionSpace(mesh, 'P', 1, dim=2)
    P1 = FiniteElement('P', tetrahedron, 1)
    element = P1 * P1 #vector element
    V = FunctionSpace(mesh, element)

    # Initial condition
    # u0 = Expression(('0.0','1.0'), degree=0)
    # u0 = project(u0, V)
    u0 = Constant((0.0, 1.0)) # π/2-pulse
    # u0 = interpolate(u0, V)

    return u0, V, mesh

def create_bt_gamma(V, mesh, B0 = -3.0, theta_deg = 90.0, a = 250):
    # ------------------------------------------------------------------------ #
    # Calculate dephasing frequency ω
    # ------------------------------------------------------------------------ #
    gamma = 2.67515255e8 # gyromagnetic ratio [rad/(s*Tesla)]

    # Susceptibility difference in blood vs. tissue including contrast agent
    CA = 0.0
    dChi_CA_per_mM = 0.3393e-6
    dChi_Blood_CA = CA * dChi_CA_per_mM

    # Susceptibilty of blood relative to tissue due to blood oxygenation and
    # hematocrit concentration is given by:
    #   deltaChi_Blood_Tissue  :=   Hct * (1-Y) * 2.26e-6 [T/T]
    Y = 0.61
    Hct = 0.44
    dChi_Blood_Oxy = 2.26e-6*Hct*(1-Y)

    # Total susceptibility difference in blood vs. tissue
    dChi_Blood = dChi_Blood_Oxy + dChi_Blood_CA

    # Dephasing frequency ω and decay rate R₂ (uniform cylinder):
    #   reference for ω: http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2962550/
    theta_rad = (np.pi/180.0)*theta_deg
    sinsqtheta = np.sin(theta_rad)**2
    # outside vessel:
    w = Expression("(0.5*gamma*B0*chi*sinsqtheta) * (a*a) * (x[1]*x[1]-x[0]*x[0]) / pow(x[0]*x[0]+x[1]*x[1],2.0)",
                    degree = 1, gamma = gamma, B0 = B0, chi = dChi_Blood, sinsqtheta = sinsqtheta, a = a)
    # inside vessel:
    # w = Expression("(chi*gamma*B0/6) * ((3.0*x[2]*x[2])/(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]) - 1.0)",
    #             degree = 1, chi = dChi_Blood, gamma = gamma, B0 = B0)

    # Vw = FunctionSpace(mesh, 'P', 1)
    # w = interpolate(w, Vw)
    # print('max(w) = ', w.vector().get_local().max())
    # print('min(w) = ', w.vector().get_local().min())

    # ------------------------------------------------------------------------ #
    # Calculate relaxation rate r
    # ------------------------------------------------------------------------ #
    T2_Tissue_Base = 69 # standard value [ms]
    if B0 == -3.0:
        T2_Tissue_Base = 69 # +/- 3 [ms]
    if B0 == -7.0:
        T2_Tissue_Base = 45.9 # +/-1.9 [ms]

    R2_Tissue_Base = 1000/T2_Tissue_Base # [ms] -> [Hz]

    r = Constant(R2_Tissue_Base)

    return w, r

def print_u(U, S0=1.0):
    # print the magnetization vector U = (u, v)
    # u, v = U.split(deepcopy=True)
    # u, v = u.vector().get_local(), v.vector().get_local()
    # u_magn = np.sqrt(np.square(u) + np.square(v))
    #
    # print('  u range: [', u.min(), ', ', u.max(), ']')
    # print('  v range: [', v.min(), ', ', v.max(), ']')
    #
    # print('||u|| range: [', np.min(u_magn), ', ', np.max(u_magn), ']')
    # print('||u|| mean:   ', np.mean(u_magn))

    # u, v = U.split(deepcopy=true)
    # print('u range: [', u.vector().get_local().min(), ', ', u.vector().get_local().max(), ']')
    # print('v range: [', v.vector().get_local().min(), ', ', v.vector().get_local().max(), ']')

    u, v = U.split()
    Sx = assemble(u*dx)
    Sy = assemble(v*dx)
    S = np.sqrt(Sx**2 + Sy**2)

    print('  [Sx, Sy] = [', Sx/S0, ', ', Sy/S0, ']')
    # print('        S  =  ', S)

    return

def bt_bwdeuler(t0 = 0.0, T = 40.0e-3, dt = 1e-3, D = 3037.0,
                save = False, foldname = 'be/tmp'):
    # Total function time
    funcstart = timer()

    # Number of timesteps
    t = t0
    tsteps = int(round(T/dt))

    # Create geometry and initial condition
    u0, V, mesh = create_bt_problem()
    omega, r2decay = create_bt_gamma(V, mesh, B0 = -3.0)

    # Define the variational problem for the Backward Euler scheme
    # U = TrialFunction(V)
    # v = TestFunction(V)
    # U_1, U_2 = split(U)
    # v_1, v_2 = split(v)
    W = TrialFunction(V)
    Z = TestFunction(V)
    u, v = split(W)
    x, y = split(Z)

    # Bloch-Torrey operator
    B = (u + r2decay*dt*u - omega*dt*v)*x*dx + \
        (v + r2decay*dt*v + omega*dt*u)*y*dx + \
        D*dt*dot(grad(u),grad(x))*dx + \
        D*dt*dot(grad(v),grad(y))*dx

    # Assmble Bloch-Torrey solver
    A = assemble(B)
    # solver = KrylovSolver(A,'bicgstab','ilu')
    solver = KrylovSolver(A,'gmres','ilu')
    solver.parameters.absolute_tolerance = 1E-7
    solver.parameters.relative_tolerance = 1E-4
    solver.parameters.maximum_iterations = 1000
    solver.parameters.nonzero_initial_guess = True

    # Export the initial data
    U = Function(V, name='Magnetization')
    U.assign(u0)

    # Compute initial signal
    u, v = U.split()
    Sx = assemble(u*dx)
    Sy = assemble(v*dx)
    S0 = np.sqrt(Sx**2 + Sy**2)

    if prnt:
        print('Step = ', 0, '/', tsteps , 'Time =', t0)
        print_u(U, S0=S0)

    if save:
        # check if folder exists
        if not os.path.exists(foldname):
            os.makedirs(foldname)

        # write signal
        signal = csv.writer(open(foldname + '/' + 'signal.csv', 'w'))
        signal.writerow([t] + [Sx] + [Sy] + [S0] + [timer()-funcstart])

        # Create VTK files for visualization output and save initial state
        # vtkfile_u = File(foldname + '/' + 'u.pvd')
        # vtkfile_v = File(foldname + '/' + 'v.pvd')
        # u, v = U.split()
        # vtkfile_u << (u, t0)
        # vtkfile_v << (v, t0)

    # Time-stepping
    for k in range(tsteps):
        # start loop time
        loopstart = timer()

        # Current time
        t = t0 + (k+1)*dt

        # Assemble the right hand side with data from current step
        u0, v0 = U.split()
        L = u0*x*dx + v0*y*dx
        b = assemble(L)
        # bc.apply(b)

        # solve into U (VectorFunction)
        solver.solve(U.vector(), b)

        # Calculate signal
        u, v = U.split()
        Sx = assemble(u*dx)
        Sy = assemble(v*dx)
        S = np.sqrt(Sx**2 + Sy**2)

        # stop loop time
        loopstop = timer()
        looptime = loopstop - loopstart

        if prnt:
            print('Step = ', k+1, '/', tsteps , 'Time =', t, 'Loop = ', looptime)
            print_u(U, S0=S0)

        if save:
            signal.writerow([t] + [Sx] + [Sy] + [S] + [timer()-funcstart])
            # vtkfile_u << (u, t)
            # vtkfile_v << (v, t)

    # Return Solution
    return U

def bt_trbdf2(t0 = 0.0, T = 40.0e-3, dt = 1e-3, D = 3037.0,
                prnt = True, save = False, foldname = 'trbdf2/tmp'):
    # Total function time
    funcstart = timer()

    # Number of timesteps
    t = t0
    tsteps = int(round(T/dt))

    # Create geometry and initial condition
    U0_vec, V, mesh = create_bt_problem()
    omega, r2decay = create_bt_gamma(V, mesh, B0 = -3.0)
    # u0, v0 = U0_vec.split() # U0_vec is the vector initial condition

    # Misc. constants for the TRBDF2 method with α = 2-√2
    c1 = (1.0 - 1.0/sqrt(2.0));
    c2 = 0.5*(1.0 + sqrt(2.0));
    c3 = 0.5*(1.0 - sqrt(2.0));

    # Define the variational problem
    W = TrialFunction(V)
    Z = TestFunction(V)
    u, v = split(W)
    x, y = split(Z)

    A = (D*dot(grad(u), grad(x)) + r2decay*u*x - omega*u*y)*dx + \
        (D*dot(grad(v), grad(y)) + r2decay*v*y + omega*v*x)*dx
    M = u*x*dx + v*y*dx
    B = M + (c1*dt)*A # LHS of the 1st and 2nd equation

    # Assemble the LHS
    A = assemble(B)
    # bc.apply(A)

    # Create solver objects (positive definite, but NON-symmetric)
    #  -> list_linear_solver_methods()
    #  -> list_krylov_solver_preconditioners()
    # solver = KrylovSolver(A,'gmres','petsc_amg') # 3.14s/loop
    solver = KrylovSolver(A,'gmres','ilu') # 1.95s/loop
    # solver = KrylovSolver(A,'gmres','sor') # 2.0s/loop
    # solver = KrylovSolver(A,'gmres','hypre_amg') # 2.05s/loop
    # solver = KrylovSolver(A,'bicgstab','ilu') # 1.94s/loop
    # solver = KrylovSolver(A,'bicgstab','sor') # 2.15s/loop
    # solver = KrylovSolver(A,'bicgstab','petsc_amg') # 3.14s/loop
    # solver = KrylovSolver(A,'bicgstab','hypre_amg') #2.16s/loop
    solver.parameters.absolute_tolerance = 1E-7
    solver.parameters.relative_tolerance = 1E-4
    solver.parameters.maximum_iterations = 1000
    solver.parameters.nonzero_initial_guess = True

    # Set initial data
    U = Function(V, name='Magnetization')
    Ua = Function(V, name='InterMag')
    U0 = Function(V, name='InitMag')
    U.assign(U0_vec)
    Ua.assign(U0_vec)
    U0.assign(U0_vec)

    # Compute initial signal
    u0, v0 = U0.split()
    Sx = assemble(u0*dx)
    Sy = assemble(v0*dx)
    S0 = np.sqrt(Sx**2 + Sy**2)

    if prnt:
        print('Step = ', 0, '/', tsteps , 'Time =', t0)
        print_u(U0, S0=S0)

    # Write initial data to file
    if save:
        if not os.path.exists(foldname):
            os.makedirs(foldname)

        # Save signal to csv-file
        signal = csv.writer(open(foldname + '/' + 'signal.csv', 'w'))
        signal.writerow([t] + [Sx] + [Sy] + [S0] + [timer()-funcstart])

        # Create VTK files for visualization output and save initial state
        # vtkfile_u = File(foldname + '/' + 'u.pvd')
        # vtkfile_v = File(foldname + '/' + 'v.pvd')
        # vtkfile_u << (u0, t0)
        # vtkfile_v << (v0, t0)

    # Time stepping
    for k in range(tsteps):
        # start loop time
        loopstart = timer()

        # Current time
        t = t0 + (k+1)*dt

        # System for the intermediate magnetization
        u0, v0 = U0.split()
        AU0 = (D*dot(grad(u0), grad(x)) + r2decay*u0*x - omega*u0*y)*dx + \
              (D*dot(grad(v0), grad(y)) + r2decay*v0*y + omega*v0*x)*dx
        MU0 = u0*x*dx + v0*y*dx
        b = assemble(MU0 - (c1*dt)*AU0)
        # bc.apply(b)

        Ua.assign(U0)
        solver.solve(Ua.vector(), b) # solve into Ua (VectorFunction)

        # System for the magnetization at the next step
        ua, va = Ua.split()
        MUa = ua*x*dx + va*y*dx
        b = assemble(c2*MUa + c3*MU0)
        # bc.apply(b)

        U.assign(Ua)
        solver.solve(U.vector(), b) # solve into U (VectorFunction)

        # Compute signal
        u, v = U.split()
        Sx = assemble(u*dx)
        Sy = assemble(v*dx)
        S = np.sqrt(Sx**2 + Sy**2)

        # stop loop time
        loopstop = timer()
        looptime = loopstop - loopstart

        if prnt:
            print('Step = ', k+1, '/', tsteps , 'Time =', t, 'Loop = ', looptime)
            print_u(U, S0=S0)

        # Write data to file
        if save:
            signal.writerow([t] + [Sx] + [Sy] + [S] + [timer()-funcstart])
            # vtkfile_u << (u, t)
            # vtkfile_v << (v, t)

        # Update
        U0.assign(U)

# ---------------------------------------------------------------------------- #
# Solve the Bloch-Torrey equation with linear finite elements, time-stepping
# using the backward euler method
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    # u = bt_bwdeuler(dt = 8e-3, save = True, foldname = 'be/tmp/dt_8e-3')
    # u = bt_bwdeuler(dt = 4e-3, save = True, foldname = 'be/tmp/dt_4e-3')
    # u = bt_bwdeuler(dt = 2e-3, save = True, foldname = 'be/tmp/dt_2e-3')
    # u = bt_bwdeuler(dt = 1e-3, save = True, foldname = 'be/tmp/dt_1e-3')
    # u = bt_bwdeuler(dt = 5e-4, save = True, foldname = 'be/tmp/dt_5e-4')
    # u = bt_bwdeuler(dt = 2.5e-4, save = True, foldname = 'be/tmp/dt_2p5e-4')
    # u = bt_bwdeuler(dt = 1.25e-4, save = True, foldname = 'be/tmp/dt_1p25e-4')
    # u = bt_bwdeuler(dt = 6.25e-5, save = True, foldname = 'be/tmp/dt_6p25e-5')
    # u = bt_bwdeuler(dt = 3.125e-5, save = True, foldname = 'be/tmp/dt_3p125e-5')

    # u = bt_trbdf2(dt = 8e-3, save = True, foldname = 'trbdf2/tmp/dt_8e-3')
    # u = bt_trbdf2(dt = 4e-3, save = True, foldname = 'trbdf2/tmp/dt_4e-3')
    # u = bt_trbdf2(dt = 2e-3, save = True, foldname = 'trbdf2/tmp/dt_2e-3')
    # u = bt_trbdf2(dt = 1e-3, save = True, foldname = 'trbdf2/tmp/dt_1e-3')
    # u = bt_trbdf2(dt = 5e-4, save = True, foldname = 'trbdf2/tmp/dt_5e-4')
    # u = bt_trbdf2(dt = 2.5e-4, save = True, foldname = 'trbdf2/tmp/dt_2p5e-4')
    # u = bt_trbdf2(dt = 1.25e-4, save = True, foldname = 'trbdf2/tmp/dt_1p25e-4')
    # u = bt_trbdf2(dt = 6.25e-5, save = True, foldname = 'trbdf2/tmp/dt_6p25e-5')

    # create_bt_problem(N = 10, nslice = 8, L = 3000.0, vesselrad = 250.0, vesselunion = True)
    # create_bt_problem(N = 25, nslice = 16, L = 3000.0, vesselrad = 250.0, vesselunion = True)
    create_bt_problem(N = 50, nslice = 32, L = 3000.0, vesselrad = 250.0, vesselunion = True)
    create_bt_problem(N = 100, nslice = 32, L = 3000.0, vesselrad = 250.0, vesselunion = True)
    create_bt_problem(N = 200, nslice = 32, L = 3000.0, vesselrad = 250.0, vesselunion = True)

    # create_bt_problem(N = 10, nslice = 8, L = 3000.0, vesselrad = 250.0, vesselunion = False)
    # create_bt_problem(N = 25, nslice = 16, L = 3000.0, vesselrad = 250.0, vesselunion = False)
    # create_bt_problem(N = 50, nslice = 32, L = 3000.0, vesselrad = 250.0, vesselunion = False)
    # create_bt_problem(N = 100, nslice = 32, L = 3000.0, vesselrad = 250.0, vesselunion = False)
    create_bt_problem(N = 200, nslice = 32, L = 3000.0, vesselrad = 250.0, vesselunion = False)
