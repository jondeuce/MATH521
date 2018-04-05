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

def get_bt_geom(loadmesh = True, savemesh = True, N = 100, nslice = 32,
    L = 3000.0, vesselrad = 250.0, vesselunion = False):
    # voxel parameters (one major vessel)
    #    L          [3000.0] voxel width [um]
    #    vesselrad  [250.0]  major vessel radius [um]
    #    N          [100]    approx. num. cells along box edge (mesh resolution)
    #    nslice     [32]     num. edges used to construct cylinder boundary


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
    P1 = FiniteElement('P', tetrahedron, 1)
    element = MixedElement([P1, P1])
    V = FunctionSpace(mesh, element)

    return V, mesh, mesh_str

def create_bt_gamma(V, mesh, B0 = -3.0, theta_deg = 90.0, a = 250.0, force_outer = True):
    # Problem parameters
    CA = 0.0 # [mM]
    Y = 0.61 # [fraction]
    Hct = 0.44 # [fraction]
    dChi_CA_per_mM = 0.3393e-6 #[(T/T)/mM]
    dR2_CA_per_mM = 5.2 # [Hz/mM]

    # ------------------------------------------------------------------------ #
    # Calculate dephasing frequency ω
    # ------------------------------------------------------------------------ #
    gamma = 2.67515255e8 # gyromagnetic ratio [rad/(s*Tesla)]
    theta_rad = np.pi * (theta_deg/180.0) # angle in radians

    # Susceptibility difference in blood vs. tissue including contrast agent
    dChi_Blood_CA = dChi_CA_per_mM * CA

    # Susceptibilty of blood relative to tissue due to blood oxygenation and
    # hematocrit concentration is given by:
    #   deltaChi_Blood_Tissue  :=   Hct * (1-Y) * 2.26e-6 [T/T]
    dChi_Blood_Oxy = 2.26e-6 * Hct * (1-Y)

    # Total susceptibility difference in blood vs. tissue
    dChi_Blood = dChi_Blood_Oxy + dChi_Blood_CA

    # Dephasing frequency ω and decay rate R₂ (uniform cylinder):
    #   reference for ω: http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2962550/
    OmegaCode = '''
    class Omega : public Expression
    {
    public:
        double gamma, B0, chi, sinsqtheta, cossqtheta, radsq;
        bool force_outer;

        Omega() : Expression() {}

        void eval(Array<double>& values, const Array<double>& x,
                  const ufc::cell& c) const
        {
            const double a2 = radsq;
            const double Kouter = gamma*B0*chi*sinsqtheta/2.0;
            const double Kinner = gamma*B0*chi*(3.0*cossqtheta-1.0)/6.0;

            const double y2x2 = x[1]*x[1] - x[0]*x[0]; // y^2 - x^2
            const double r2 = x[0]*x[0] + x[1]*x[1]; // x^2 + y^2
            const double InnerVal = Kinner;
            const double OuterVal = Kouter * (a2/r2) * (y2x2/r2);

            if (force_outer) {
                // Assume all points are outside
                values[0] = OuterVal;
            } else {
                // Points may be inside
                values[0] = r2 >= a2 ? OuterVal : InnerVal;
            }

            return;
        }
    };'''

    # Code expression
    w = Expression(OmegaCode, degree = 1, force_outer = force_outer,
        gamma = gamma, B0 = B0, chi = dChi_Blood, radsq = a**2,
        sinsqtheta = np.sin(theta_rad)**2, cossqtheta = np.cos(theta_rad)**2)

    # ------------------------------------------------------------------------ #
    # Calculate relaxation rate r
    # ------------------------------------------------------------------------ #
    T2_Tissue_Base = 69.0 # B0 = -3.0T value [ms]
    dR2_Blood_Oxy = 30.0125 # B0 = -3.0T value (with Y = 0.61) [Hz]

    if B0 == -3.0:
        T2_Tissue_Base = 69.0 # +/- 3 [ms]
        dR2_Blood_Oxy = 30.0125 # For Y = 0.61 [Hz]
    if B0 == -7.0:
        T2_Tissue_Base = 45.9 # +/-1.9 [ms]
        dR2_Blood_Oxy = 71.114475 # For Y = 0.61 [Hz]

    dR2_Blood_CA = dR2_CA_per_mM * CA # R2 change due to CA [ms]
    R2_Blood = dR2_Blood_Oxy + dR2_Blood_CA # Total R2 [ms]
    R2_Tissue = 1000.0/T2_Tissue_Base # [ms] -> [Hz]

    R2DecayCode = '''
    class R2Decay : public Expression
    {
    public:
        double R2_Blood, R2_Tissue, radsq;

        R2Decay() : Expression() {}

        void eval(Array<double>& values, const Array<double>& x,
                  const ufc::cell& c) const
        {
            // Points may be inside
            const double a2 = radsq;
            const double r2 = x[0]*x[0] + x[1]*x[1]; // x^2 + y^2
            values[0] = r2 >= a2 ? R2_Tissue : R2_Blood;

            return;
        }
    };'''

    if force_outer:
        r = Constant(R2_Tissue)
    else:
        r = Expression(R2DecayCode, degree = 1, R2_Tissue = R2_Tissue, R2_Blood = R2_Blood)

    return w, r

def bt_bilinear(U,Z,D,r,w):
    # Weak form of the Bloch-Torrey operator. Output operators are defined
    # such that we have:
    #   M*dU/dt = -A*U
    #
    # U is a two-element vector, Z is a two-element vector, D is the diffusion
    # coefficient, r is the decay rate, and w is the precession rate

    u, v = split(U) # "trial function" -> components
    x, y = split(Z) # test function -> components

    # Weak form bloch torrey operator:
    #   A = [x, y] * [-DΔ+r    -w] * [u] dx
    #                [   w  -DΔ+r]   [v]
    A = dot(D*grad(u),grad(x))*dx + (r*u - w*v)*x*dx + \
        dot(D*grad(v),grad(y))*dx + (r*v + w*u)*y*dx

    return A

def M_bilinear(U,Z):
    # Return the mass matrix operator: M = dot(U,Z)*dx
    u, v = split(U) # "trial function" -> components
    x, y = split(Z) # test function -> components
    M = (u*x + v*y)*dx
    return M

def S_signal(U):
    u, v = U.split()
    Sx = assemble(u*dx) # x-component of signal
    Sy = assemble(v*dx) # y-component of signal
    S = np.hypot(Sx, Sy)
    return Sx, Sy, S

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

    Sx, Sy, S = S_signal(U)
    print('  [Sx, Sy] = [', Sx/S0, ', ', Sy/S0, ']')
    # print('        S  =  ', S)

    return

def bt_bwdeuler(V, mesh, omega, r2decay,
                t0 = 0.0, T = 40.0e-3, dt = 1.0e-3, Dcoeff = 3037.0, B0 = -3.0,
                prnt = True, save = False, foldname = 'be/tmp'):
    # Total function time
    funcstart = timer()

    # Number of timesteps
    t = t0
    tsteps = int(round(T/dt))

    # Define the variational problem for the Backward Euler scheme
    W = TrialFunction(V)
    Z = TestFunction(V)

    # Bloch-Torrey operator (backward euler step):
    #   dU/dt = -A*U => (M + A*dt)U = M*U0
    A = bt_bilinear(W,Z,Dcoeff,r2decay,omega)
    M = M_bilinear(W,Z)
    B = M + dt*A

    # Assmble Bloch-Torrey solver
    B_mat = assemble(B)

    # solver = KrylovSolver(B_mat,'bicgstab','ilu')
    solver = KrylovSolver(B_mat,'gmres','ilu')
    solver.parameters.absolute_tolerance = 1E-7
    solver.parameters.relative_tolerance = 1E-4
    solver.parameters.maximum_iterations = 1000
    solver.parameters.nonzero_initial_guess = True

    # Export the initial data
    U = Function(V, name='Magnetization')

    U0_vec = Constant((0.0, 1.0)) # π/2-pulse
    U.assign(U0_vec)

    # Compute initial signal
    Sx0, Sy0, S0 = S_signal(U)

    if prnt:
        print('Step = ', 0, '/', tsteps , 'Time =', t0)
        print_u(U, S0=S0)

    if save:
        # check if folder exists
        if not os.path.exists(foldname):
            os.makedirs(foldname)

        # write signal
        signal = csv.writer(open(foldname + '/' + 'signal.csv', 'w'))
        signal.writerow([t] + [Sx0] + [Sy0] + [S0] + [timer()-funcstart])

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

        # Assemble the right hand (backward euler step; U = U0 here)
        #   dU/dt = -A*U => (M + A*dt)U = M*U0
        L = M_bilinear(U,Z)
        b = assemble(L)
        # bc.apply(b)

        # solve into U (VectorFunction)
        solver.solve(U.vector(), b)

        # Calculate signal
        Sx, Sy, S = S_signal(U)

        # stop loop time
        loopstop = timer()
        looptime = loopstop - loopstart

        if prnt:
            print('Step = ', k+1, '/', tsteps , 'Time =', t, 'Loop = ', looptime)
            print_u(U, S0=S0)

        if save:
            signal.writerow([t] + [Sx] + [Sy] + [S] + [timer()-funcstart])
            # u, v = U.split()
            # vtkfile_u << (u, t)
            # vtkfile_v << (v, t)

    return U

def bt_trbdf2(V, mesh, omega, r2decay,
              t0 = 0.0, T = 40.0e-3, dt = 1.0e-3, Dcoeff = 3037.0, B0 = -3.0,
              prnt = True, save = False, foldname = 'trbdf2/tmp'):
    # Total function time
    funcstart = timer()

    # Number of timesteps
    t = t0
    tsteps = int(round(T/dt))

    # Misc. constants for the TRBDF2 method with α = 2-√2
    c0 = (1.0 - 1.0/sqrt(2.0));
    c1 = 0.5*(1.0 + sqrt(2.0));
    c2 = 0.5*(1.0 - sqrt(2.0));

    # Define the variational problem
    W = TrialFunction(V)
    Z = TestFunction(V)

    # LHS of both TRBDF2 steps (α = 2-√2):
    #   (M+c0*dt*A)*Ua = (M - c0*dt*A)*U0
    #   (M+c0*dt*A)*U  = c1*M*Ua + c2*M*U0
    A = bt_bilinear(W,Z,Dcoeff,r2decay,omega)
    M = M_bilinear(W,Z)
    B = M + (c0*dt)*A # LHS of the 1st and 2nd equation

    # Create solver objects (positive definite, but NON-symmetric)
    #  -> list_linear_solver_methods()
    #  -> list_krylov_solver_preconditioners()
    B_mat = assemble(B)
    # bc.apply(A)

    # solver = KrylovSolver(B_mat,'gmres','petsc_amg') # 3.14s/loop
    solver = KrylovSolver(B_mat,'gmres','ilu') # 1.95s/loop
    # solver = KrylovSolver(B_mat,'gmres','sor') # 2.0s/loop
    # solver = KrylovSolver(B_mat,'gmres','hypre_amg') # 2.05s/loop
    # solver = KrylovSolver(B_mat,'bicgstab','ilu') # 1.94s/loop
    # solver = KrylovSolver(B_mat,'bicgstab','sor') # 2.15s/loop
    # solver = KrylovSolver(B_mat,'bicgstab','petsc_amg') # 3.14s/loop
    # solver = KrylovSolver(B_mat,'bicgstab','hypre_amg') #2.16s/loop
    solver.parameters.absolute_tolerance = 1E-7
    solver.parameters.relative_tolerance = 1E-4
    solver.parameters.maximum_iterations = 1000
    solver.parameters.nonzero_initial_guess = True

    # Set initial data
    U = Function(V, name='Magnetization')
    Ua = Function(V, name='InterMag')
    U0 = Function(V, name='InitMag')

    U0_vec = Constant((0.0, 1.0)) # π/2-pulse
    U0.assign(U0_vec)

    # Compute initial signal
    Sx0, Sy0, S0 = S_signal(U0)

    if prnt:
        print('Step = ', 0, '/', tsteps , 'Time =', t0)
        print_u(U0, S0=S0)

    # Write initial data to file
    if save:
        if not os.path.exists(foldname):
            os.makedirs(foldname)

        # Save signal to csv-file
        signal = csv.writer(open(foldname + '/' + 'signal.csv', 'w'))
        signal.writerow([t] + [Sx0] + [Sy0] + [S0] + [timer()-funcstart])

        # Create VTK files for visualization output and save initial state
        # vtkfile_u = File(foldname + '/' + 'u.pvd')
        # vtkfile_v = File(foldname + '/' + 'v.pvd')
        # u0, v0 = U0.split()
        # vtkfile_u << (u0, t0)
        # vtkfile_v << (v0, t0)

    # Time stepping
    for k in range(tsteps):
        # start loop time
        loopstart = timer()

        # Current time
        t = t0 + (k+1)*dt

        # RHS of second TRBDF2 step (α = 2-√2):
        #   (M+c0*dt*A)*Ua = M*U0 - c0*dt*A*U0
        A_U0 = bt_bilinear(U0,Z,Dcoeff,r2decay,omega)
        M_U0 = M_bilinear(U0,Z)
        b = assemble(M_U0 - (c0*dt)*A_U0)
        # bc.apply(b)

        Ua.assign(U0)
        solver.solve(Ua.vector(), b) # solve into Ua (VectorFunction)

        # RHS of second TRBDF2 step (α = 2-√2):
        #   (M+c0*dt*A)*U  = c1*M*Ua + c2*M*U0z
        M_Ua = M_bilinear(Ua,Z)
        b = assemble(c1*M_Ua + c2*M_U0)
        # bc.apply(b)

        U.assign(Ua)
        solver.solve(U.vector(), b) # solve into U (VectorFunction)

        # Compute signal
        Sx, Sy, S = S_signal(U)

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

    return U

# ---------------------------------------------------------------------------- #
# Solve the Bloch-Torrey equation with linear finite elements, time-stepping
# using the backward euler method
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    # vesselunionlist = [True, False]
    vesselunionlist = [True]

    # Nlist, nslicelist = [10,25,50,100], [8,16,32,32]
    # Nlist, nslicelist = [10,25,50], [8,16,32]
    # Nlist, nslicelist = [10], [8]
    Nlist, nslicelist = [200], [64]

    # Number of simulations to do, halving time step each time.
    # Minimum time step dt_min = dt0/2^(Num-1):
    #   9 -> dt_min = 8e-3/2**8 = 3.125e-5
    #   8 -> dt_min = 8e-3/2**7 = 6.25e-5
    #   7 -> dt_min = 8e-3/2**6 = 0.000125
    dt0 = 8.0e-3 # [s]
    NumBE = 0
    NumTR = 7

    vesselradius = 250.0
    theta_deg = 90.0
    B0 = -3.0

    for vesselunion in vesselunionlist:
        for N, nslice in zip(Nlist,nslicelist):

            print("\n", "vesselunion = ", vesselunion, ", N = ", N, ", nslice = ", nslice, "\n")

            Geomargs = {'N':N, 'nslice':nslice, 'vesselrad':vesselradius, 'vesselunion':vesselunion}
            Gammaargs = {'B0':B0, 'theta_deg':theta_deg, 'a':vesselradius, 'force_outer':not vesselunion}

            V, mesh, mesh_str = get_bt_geom(**Geomargs)
            w, r = create_bt_gamma(V, mesh, **Gammaargs)

            parent_foldname = 'bt/results/union' if vesselunion else 'bt/results/hollow';
            results_foldname = parent_foldname + '/' + mesh_str

            # ---------------------------------------------------------------- #
            # Backward Euler Method
            # ---------------------------------------------------------------- #
            dt = dt0
            for _ in range(NumBE):
                BEfoldname = results_foldname + '/be/dt_' + str(dt).replace('.','p')
                BEargs = {'dt':dt, 'save':True, 'foldname':BEfoldname}

                print("\n", "BE: dt = ", dt, "\n")
                bt_bwdeuler(V, mesh, w, r, **BEargs)

                dt = 0.5*dt

            # ---------------------------------------------------------------- #
            # TRBDF2 (α = 2-√2) Method
            # ---------------------------------------------------------------- #
            dt = dt0
            for _ in range(NumTR):
                TRfoldname = results_foldname + '/trbdf2/dt_' + str(dt).replace('.','p')
                TRargs = {'dt':dt, 'save':True, 'foldname':TRfoldname}

                print("\n", "TRBDF2: dt = ", dt, "\n")
                bt_trbdf2(V, mesh, w, r, **TRargs)

                dt = 0.5*dt
