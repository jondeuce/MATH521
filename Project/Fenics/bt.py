# coding=utf-8
"""
FEniCS program: Solution of the Bloch-Torrey equation with Neumann boundary
conditions.
"""

from __future__ import print_function
from fenics import *
from mshr import *
import time
import numpy as np

def create_bt_problem(loadmesh = True, savemesh = True):
    if loadmesh:
        mesh = Mesh('bt/geom/cylinder.xml.gz')
    else:
        # Create a geometry and mesh it
        Nx, Ny, Nz = 30, 30, 30
        Lx, Ly, Lz = 3000, 3000, 3000
        meshsize = ((Lx*Ly*Lz)/(Nx*Ny*Nz))**(1.0/3.0) # h ~ ∛(CellVolume)

        vesselrad = 250 # cylindrical vessel radius [um]
        vesselcircum = 2*np.pi*vesselrad
        nslice = 32 # number of edges used to construct cylinder boundary

        # voxel = BoxMesh(Point(-Lx/2, -Ly/2, -Lz/2), Point(Lx/2, Ly/2, Lz/2), Nx, Ny, Nz)
        voxel = Box(Point(-Lx/2, -Ly/2, -Lz/2), Point(Lx/2, Ly/2, Lz/2))
        vessel = Cylinder(Point(0,0,-Lz/2), Point(0,0,Lz/2), vesselrad, vesselrad, segments = nslice)
        domain = voxel - vessel

        mesh = generate_mesh(domain, meshsize)

        if savemesh:
            File('bt/geom/cylinder.xml.gz') << mesh

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
    u0 = interpolate(u0, V)

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

def print_u(u):
    # print the magnetization vector u = (u_1, u_2)
    u_1, u_2 = u.split(deepcopy=True)
    u_1, u_2 = u_1.vector().get_local(), u_2.vector().get_local()
    u_magn = np.sqrt(np.square(u_1) + np.square(u_2))

    print('  u_1 range: [', u_1.min(), ', ', u_1.max(), ']')
    print('  u_2 range: [', u_2.min(), ', ', u_2.max(), ']')

    print('||u|| range: [', np.min(u_magn), ', ', np.max(u_magn), ']')
    print('||u|| mean:   ', np.mean(u_magn))

    return

def bt_bwdeuler(t0 = 0.0, T = 40.0e-3, dt = 5e-4, D = 3037.0,
                prnt = True, save = False, foldname = 'bt/tmp'):
    # Number of timesteps
    tsteps = int(round(T/dt))

    # Create geometry and initial condition
    u0, V, mesh = create_bt_problem()
    w, r = create_bt_gamma(V, mesh, B0 = -3.0)

    # Define the variational problem for the Backward Euler scheme
    U = TrialFunction(V)
    v = TestFunction(V)
    U_1, U_2 = split(U)
    v_1, v_2 = split(v)

    # Bloch-Torrey operator
    B = (U_1 + r*dt*U_1 - w*dt*U_2)*v_1*dx + \
        (U_2 + r*dt*U_2 + w*dt*U_1)*v_2*dx + \
        D*dt*inner(grad(U_1),grad(v_1))*dx + \
        D*dt*inner(grad(U_2),grad(v_2))*dx

    # Export the initial data
    u = Function(V, name='Magnetization')
    u.assign(u0)

    if prnt:
        print('Step = ', 0, '/', tsteps , 'Time =', t0)
        print_u(u0)

    if save:
        # Create VTK files for visualization output and save initial state
        vtkfile_u_1 = File(foldname + '/' + 'u_1.pvd')
        vtkfile_u_2 = File(foldname + '/' + 'u_2.pvd')
        u_1, u_2 = u.split()
        vtkfile_u_1 << (u_1, t0)
        vtkfile_u_2 << (u_2, t0)

    # Time-stepping
    t = t0
    for k in range(tsteps):
        # Current time
        t = t0 + (k+1)*dt

        # Assemble the right hand side
        u_1, u_2 = u.split()
        L = u_1*v_1*dx + u_2*v_2*dx

        # Solve variational problem for time step
        solve(B == L, u, solver_parameters={"linear_solver": "bicgstab"})

        if prnt:
            print('Step = ', k+1, '/', tsteps , 'Time =', t)
            print_u(u)

        if save:
            u_1, u_2 = u.split()
            vtkfile_u_1 << (u_1, t)
            vtkfile_u_2 << (u_2, t)

        # Update previous solution
        # u0.assign(u)

        # Update progress bar
        # progress.update(t / T)

    # Return Solution
    return u

# ---------------------------------------------------------------------------- #
# Solve the Bloch-Torrey equation with linear finite elements, time-stepping
# using the backward euler method
# ---------------------------------------------------------------------------- #
u = bt_bwdeuler(prnt = True, save = True)
