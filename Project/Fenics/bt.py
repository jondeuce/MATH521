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

def create_bt_problem(loadmesh = False, savemesh = True):
    if loadmesh:
        mesh = Mesh('bt/cylinder.xml.gz')
    else:
        # Create a geometry and mesh it
        Nx, Ny, Nz = 30, 30, 30
        Lx, Ly, Lz = 3000, 3000, 3000
        meshsize = ((Lx*Ly*Lz)/(Nx*Ny*Nz))**(1.0/3.0) # h ~ ∛(CellVolume)

        vesselrad = 100 # cylindrical vessel radius [um]
        vesselcircum = 2*np.pi*vesselrad
        nslice = 20 # number of edges used to construct cylinder boundary

        # voxel = BoxMesh(Point(-Lx/2, -Ly/2, -Lz/2), Point(Lx/2, Ly/2, Lz/2), Nx, Ny, Nz)
        voxel = Box(Point(-Lx/2, -Ly/2, -Lz/2), Point(Lx/2, Ly/2, Lz/2))
        vessel = Cylinder(Point(0,0,-Lz/2), Point(0,0,Lz/2), vesselrad, nslice)
        domain = voxel - vessel

        mesh = generate_mesh(domain, meshsize)

        if savemesh:
            File('bt/cylinder.xml.gz') << mesh

    # Define function space
    # element = VectorElement('P', triangle, 1, dim=2)
    # V = FunctionSpace(mesh, element)
    V = VectorFunctionSpace(mesh, 'P', 1, dim=2)

    # Initial condition
    # u0 = Expression(('0.0','1.0'), degree=0)
    # u0 = project(u0, V)
    u0 = Constant((0.0, 1.0)) # π/2-pulse
    u0 = interpolate(u0, V)

    return u0, V

def create_bt_gamma(B0 = -3.0):
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
    w = Expression("(chi*gamma*B0/6) * ((3.0*x[2]*x[2])/(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]) - 1.0)",
                    degree = 1, chi = dChi_Blood, gamma = gamma, B0 = B0)

    # ------------------------------------------------------------------------ #
    # Calculate relaxation rate r
    # ------------------------------------------------------------------------ #
    if B0 == -3.0:
        T2_Tissue_Base = 69 # +/- 3 [ms]
    elif B0 == -7.0:
        T2_Tissue_Base = 45.9 # +/-1.9 [ms]

    R2_Tissue_Base = 1000/T2_Tissue_Base # [ms] -> [Hz]

    r = Constant(R2_Tissue_Base)

    return w, r

def bt_bwdeuler(t0 = 0.0, T = 40.0e-3, dt = 5e-4, D = 3037.0,
                prnt = True, save = False, foldname = 'bt'):
    # Create geometry and initial condition
    u0, V = create_bt_problem()
    w, r = create_bt_gamma()

    # Define the variational problem for the Backward Euler scheme
    u = TrialFunction(V)
    u_1, u_2 = split(u)
    v_1, v_2 = TestFunctions(V)

    # Bloch-Torrey operator
    B = (1+r*dt+w*dt)*u_1*v_1*dx + \
        (1+r*dt-w*dt)*u_2*v_2*dx + \
        D*dt*dot(grad(u_1),grad(v_1))*dx + \
        D*dt*dot(grad(u_2),grad(v_2))*dx

    # Export the initial data
    u = Function(V, name='Magnetization')
    u.assign(u0)
    if save:
        # Create VTK files for visualization output and save initial state
        vtkfile_u_1 = File(foldname + '/' + 'u_1.pvd')
        vtkfile_u_2 = File(foldname + '/' + 'u_2.pvd')
        _u_1, _u_2 = u.split()
        vtkfile_u_1 << (_u_1, t0)
        vtkfile_u_2 << (_u_2, t0)

    # Time-stepping
    t = t0
    tsteps = int(round(T/dt))
    for k in range(tsteps):
        # Current time
        t = t0 + (k+1)*dt
        if prnt:
            print('Step = ', k+1, '/', tsteps , 'Time =', t)

        # Assemble the right hand side
        u0_1, u0_2 = u0.split()
        L = u0_1*v_1*dx + u0_2*v_2*dx

        # Solve variational problem for time step
        solve(B == L, u, solver_parameters={"linear_solver": "cg"})

        if save:
            # Save solution to file (VTK)
            _u_1, _u_2 = u.split()
            vtkfile_u_1 << (_u_1, t)
            vtkfile_u_2 << (_u_2, t)

        # Update previous solution
        u0.assign(u)

        # Update progress bar
        # progress.update(t / T)

    # Return Solution
    return u

# ---------------------------------------------------------------------------- #
# Solve the Bloch-Torrey equation with linear finite elements, time-stepping
# using the backward euler method
# ---------------------------------------------------------------------------- #
u = bt_bwdeuler(t0 = 0.0, T = 40.0e-3, dt = 5e-4, D = 3037.0,
                prnt = True, save = True, foldname = 'bt')
