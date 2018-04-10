# coding=utf-8
"""
FEniCS program: Solution of the Bloch-Torrey equation with Neumann boundary
conditions.
"""

from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
import itertools
import time
import csv
import os
from timeit import default_timer as timer

# ------------------------------------------------------------------------ #
# General global constants
# ------------------------------------------------------------------------ #
B0 = -3.0
gamma = 2.67515255e8 # gyromagnetic ratio [rad/(s*Tesla)]
theta_deg = 90.0
theta_rad = np.pi * (theta_deg/180.0) # angle in radians
CA = 0.0 # [mM]
Y = 0.61 # [fraction]
Hct = 0.44 # [fraction]

# Misc. constants for the TRBDF2 method with α = 2-√2
c0 = (1.0 - 1.0/sqrt(2.0));
c1 = 0.5*(1.0 + sqrt(2.0));
c2 = 0.5*(1.0 - sqrt(2.0));

# ------------------------------------------------------------------------ #
# Global constants for calculating frequency ω
# ------------------------------------------------------------------------ #
dChi_CA_per_mM = 0.3393e-6 #[(T/T)/mM]
dChi_Blood_CA = dChi_CA_per_mM * CA # δχ due to contrast agent

# Susceptibilty of blood relative to tissue due to blood oxygenation and
# hematocrit concentration is given by:
#   dChi_Blood_Oxy := Hct * (1-Y) * 2.26e-6 [T/T]
dChi_Blood_Oxy = 2.26e-6 * Hct * (1-Y)
dChi_Blood = dChi_Blood_Oxy + dChi_Blood_CA # Total δχ in blood vs. tissue

# ------------------------------------------------------------------------ #
# Global constants for calculating relaxation rate R₂
# ------------------------------------------------------------------------ #
if abs(abs(B0)-3.0) < 5*DOLFIN_EPS:
    T2_Tissue_Base = 69.0 # B0 = ±3.0T value [ms]
    dR2_Blood_Oxy = 30.0125 # B0 = ±3.0T value (with Y = 0.61) [Hz]
elif abs(abs(B0)-7.0) < 5*DOLFIN_EPS:
    T2_Tissue_Base = 45.9 # B0 = ±7.0T value, +/-1.9 [ms]
    dR2_Blood_Oxy = 71.114475 # For Y = 0.61, Hct = 0.44 [Hz]
else:
    raise ValueError('B0 must be -3.0 or -7.0')

dR2_CA_per_mM = 5.2 # [Hz/mM]
dR2_Blood_CA = dR2_CA_per_mM * CA # R2 change due to CA [ms]
R2_Blood = dR2_Blood_Oxy + dR2_Blood_CA # Total R2 [ms]
R2_Tissue = 1000.0/T2_Tissue_Base # [ms] -> [Hz]

def get_mesh_str(N,nslice,rad):
    rad = int(round(rad))
    return 'cylinder_N' + str(N) + '_ns' + str(nslice) + '_r' + str(rad)

def get_bt_geom(loadmesh = True, savemesh = True, N = 100, nslice = 32,
    L = 3000.0, vesselrad = 250.0, isvesselunion = False):
    # voxel parameters (one major vessel)
    #    L          [3000.0] voxel width [um]
    #    vesselrad  [250.0]  major vessel radius [um]
    #    N          [100]    approx. num. cells along box edge (mesh resolution)
    #    nslice     [32]     num. edges used to construct cylinder boundary

    mesh_str = get_mesh_str(N,nslice,vesselrad)
    geom_foldname = 'bt/geom/union' if isvesselunion else 'bt/geom/hollow';
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
        vessel = Cylinder(Point(0,0,-Lz/2), Point(0,0,Lz/2),
            vesselrad, vesselrad, segments = nslice)

        if isvesselunion:
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
    elem = MixedElement([P1, P1])
    V = FunctionSpace(mesh, elem)

    return V, elem, mesh, mesh_str

def create_bt_gamma(V, mesh, a = 250.0, force_outer = True):
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
    omega = Expression(OmegaCode, degree = 1, force_outer = force_outer,
        gamma = gamma, B0 = B0, chi = dChi_Blood, radsq = a**2,
        sinsqtheta = np.sin(theta_rad)**2, cossqtheta = np.cos(theta_rad)**2)

    # ------------------------------------------------------------------------ #
    # Calculate relaxation rate r2decay
    # ------------------------------------------------------------------------ #
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
        r2decay = Constant(R2_Tissue)
    else:
        r2decay = Expression(R2DecayCode, degree = 0,
            R2_Tissue = R2_Tissue, R2_Blood = R2_Blood)

    return omega, r2decay

def bt_bilinear(U,Z,Dcoeff,r2decay,omega,isdual=False):
    # Weak form of the Bloch-Torrey operator. Output operators are defined
    # such that we have:
    #   M*dU/dt = -A*U
    #
    # U is a two-element Function or TrialFunction, Z is a two-element
    # TestFunction, Dcoeff is the diffusion coefficient, r2decay is the
    # transverse decay rate, and omega is the precession rate

    u, v = split(U) # "trial function" -> components
    w, z = split(Z) # test function -> components

    # Weak form bloch torrey operator:
    #   A = [w, z] * [-DΔ+R₂,     -ω] * [u] dx
    #                [     ω, -DΔ+R₂]   [v]
    if not isdual:
        A = dot(Dcoeff*grad(u),grad(w))*dx + (r2decay*u*w - omega*v*w)*dx + \
            dot(Dcoeff*grad(v),grad(z))*dx + (r2decay*v*z + omega*u*z)*dx
    else:
        # [u,v] <--> [w,z], or equivalently, ω --> -ω
        A = dot(Dcoeff*grad(w),grad(u))*dx + (r2decay*w*u - omega*z*u)*dx + \
            dot(Dcoeff*grad(z),grad(v))*dx + (r2decay*z*v + omega*w*v)*dx

    return A

def M_bilinear(U,Z):
    # Return the mass matrix operator: M = dot(U,Z)*dx
    u, v = split(U) # "trial function" -> components
    w, z = split(Z) # test function -> components
    M = (u*w + v*z)*dx
    return M

def S_signal(U):
    u, v = U.split()
    Sx = assemble(u*dx) # x-component of signal
    Sy = assemble(v*dx) # y-component of signal
    S = np.hypot(Sx, Sy) # magnitude of signal
    return Sx, Sy, S

def print_u(U, S0=1.0):
    Sx, Sy, S = S_signal(U)
    print('[Sx, Sy] = [', Sx/S0, ', ', Sy/S0, ']')

    return

def accum_max(acc,Flast,F,dt,k,N,Dcoeff,omega,r2decay):
    # Records the maximum of successive F's observed
    if acc is None:
        acc = 0.0
    else:
        acc = np.maximum(acc, F)

    return acc

def accum_max_normdiff(acc,Flast,F,dt,k,N,Dcoeff,omega,r2decay,norm_type='L2'):
    # Records the maximum difference in norm between successive F's
    f = 0.0 if acc is None else errornorm(Flast, F, norm_type=norm_type)
    acc = accum_max(acc,None,f,dt,k,N,Dcoeff,omega,r2decay)

    return acc

def accum_max_vecnormdiff(acc,Flast,F,dt,k,N,Dcoeff,omega,r2decay):
    # Records the maximum difference in vector norm between successive F's
    f = 0.0 if acc is None else np.linalg.norm(F.vector()-Flast.vector())
    acc = accum_max(acc,None,f,dt,k,N,Dcoeff,omega,r2decay)

    return acc

def accum_int_Function(acc,Flast,F,dt,k,N,Dcoeff,omega,r2decay):
    # Integrates F using the trapezoidal rule
    if acc is None:
        # Initialize accumulator; F is the inital signal U0
        acc = Function(F.function_space())
        acc.vector()[:] = (0.5*dt)*F.vector()
    elif k+1 == N:
        acc.vector()[:] += (0.5*dt)*F.vector()
    else:
        acc.vector()[:] += dt*F.vector()

    return acc

def accum_int_Scalar(acc,Flast,F,dt,k,N,Dcoeff,omega,r2decay):
    # Integrates F using the trapezoidal rule
    if acc is None:
        # Initialize accumulator; F is the inital signal U0
        acc = 0.0
        acc = (0.5*dt)*F
    elif k+1 == N:
        acc += (0.5*dt)*F
    else:
        acc += dt*F

    return acc

def accum_dual(acc,Ulast,U,dt,k,N,Dcoeff,omega,r2decay):
    # Integrates the square of the dual solution derivative times the mass
    # matrix, i.e. the RHS of: M*dU/dt = -A*U
    Z = TestFunction(U.function_space())
    dU = Function(U.function_space())

    At_U = bt_bilinear(U,Z,Dcoeff,r2decay,omega,isdual=True)
    dU.vector()[:] = assemble(-At_U) # M*dU/dt = -A*U

    if acc is None:
        acc1 = acc2 = None
    else:
        acc1, acc2 = acc

    dUsq = Function(dU.function_space())
    dUsq.vector()[:] = np.square(dU.vector().get_local())
    acc1 = accum_int_Function(acc1,None,dUsq,dt,k,N,Dcoeff,omega,r2decay)
    # acc1 = accum_int_square(acc,None,dU,dt,k,N,Dcoeff,omega,r2decay)

    dUnorm = np.sqrt(np.sum(dUsq.vector().get_local()))
    acc2 = accum_int_Scalar(acc2,None,dUnorm,dt,k,N,Dcoeff,omega,r2decay)

    acc = (acc1, acc2)
    return acc

def get_dual_init(psi, V):
    U = TrialFunction(V)
    Z = TestFunction(V)

    M_form = M_bilinear(U,Z)
    M_mat = assemble(M_form)

    # M_lumped = M_mat
    # M_lumped.zero()
    # M_action_form = action(M_form, Constant(1.0))
    # M_lumped.set_diagonal(assemble(M_action_form))

    solver = KrylovSolver(M_mat,'cg','icc') # 1.95s/loop
    solver.parameters.absolute_tolerance = 1E-12
    solver.parameters.relative_tolerance = 1E-10
    solver.parameters.maximum_iterations = 1000
    solver.parameters.nonzero_initial_guess = False

    # solve M*phi = psi
    phi = Function(V, name="InitDualMag")
    solver.solve(phi.vector(),psi.vector())

    return phi

def bt_solver(U0, V, mesh, Dcoeff, omega, r2decay,
              t0 = 0.0, T = 40.0e-3, dt = 1.0e-3,
              stepper = 'be', accumulator = None, isdual = False,
              prnt = True, savesignal = False, savemag = False,
              foldname = 'solver/tmp'):
    # Total function time
    funcstart = timer()

    # Number of timesteps
    t = t0
    tsteps = int(round(T/dt))

    # Define the variational problem
    W = TrialFunction(V)
    Z = TestFunction(V)
    A = bt_bilinear(W,Z,Dcoeff,r2decay,omega,isdual=isdual)
    M = M_bilinear(W,Z)

    # Assemble LHS matrix form B
    if stepper.lower() == 'trbdf2':
        # TRBDF2 with α = 2-√2
        B = M + (c0*dt)*A
    else:
        # Backward Euler
        B = M + dt*A

    B_mat = assemble(B) # bc.apply(B_mat), if there were BC's to set

    # Create solver objects (positive definite, but NON-symmetric)
    #  -> list_linear_solver_methods()
    #  -> list_krylov_solver_preconditioners()

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

    # Method specific initialization (temp variables, etc.)
    if stepper.lower() == 'trbdf2':
        Ua = Function(V, name='InterMag')
    else: #'be'
        U.assign(U0)

    # Compute initial signal
    Sx0, Sy0, S0 = S_signal(U0)

    if prnt:
        print('Step = ', 0, '/', tsteps , 'Time =', t0)
        print_u(U0, S0=S0)

    # Write initial data to file
    if savesignal:
        if not os.path.exists(foldname):
            os.makedirs(foldname)

        # Save signal to csv-file
        signal = csv.writer(open(foldname + '/' + 'signal.csv', 'w'))
        signal.writerow([t] + [Sx0] + [Sy0] + [S0] + [timer()-funcstart])

    if savemag:
        # Create VTK files for visualization output and save initial state
        vtkfile_u = File(foldname + '/mag/u.pvd')
        vtkfile_v = File(foldname + '/mag/v.pvd')
        u0, v0 = U0.split()
        vtkfile_u << (u0, t0)
        vtkfile_v << (v0, t0)

    if accumulator is None:
        acc = None
    else:
        acc = accumulator(None,None,U0,dt,None,tsteps,Dcoeff,omega,r2decay)

    # Time stepping
    for k in range(tsteps):
        # start loop time
        loopstart = timer()

        # Current time
        t = t0 + (k+1)*dt

        # Assemble rhs vector form L
        if stepper.lower() == 'trbdf2':
            # TRBDF2 steps (α = 2-√2, dU/dt = -A*U):
            #   (M+c0*dt*A)*Ua = (M - c0*dt*A)*U0
            #   (M+c0*dt*A)*U  = c1*M*Ua + c2*M*U0

            A_U0 = bt_bilinear(U0,Z,Dcoeff,r2decay,omega,isdual=isdual)
            M_U0 = M_bilinear(U0,Z)
            L = M_U0 - (c0*dt)*A_U0
            b = assemble(L)
            # bc.apply(b)

            # Solve first step:
            Ua.assign(U0)
            solver.solve(Ua.vector(), b) # solve into Ua (VectorFunction)
            U.assign(Ua)

            # Set up RHS of second step:
            M_Ua = M_bilinear(Ua,Z)
            L = c1*M_Ua + c2*M_U0
        else:
            # Backward Euler step (dU/dt = -A*U):
            #   (M + A*dt)U = M*U0
            L = M_bilinear(U0,Z)

        # Solve into U (VectorFunction)
        b = assemble(L) # bc.apply(b), if there were BC's to set
        solver.solve(U.vector(), b)

        # Compute signal
        Sx, Sy, S = S_signal(U)

        # stop loop time
        loopstop = timer()
        looptime = loopstop - loopstart

        if prnt:
            print('Step = ', k+1, '/', tsteps , 'Time =', t, 'Loop = ', looptime)
            print_u(U, S0=S0)

        # Write data to file
        if savesignal:
            signal.writerow([t] + [Sx] + [Sy] + [S] + [timer()-funcstart])

        if savemag:
            u, v = U.split()
            vtkfile_u << (u, t)
            vtkfile_v << (v, t)

        if not (accumulator is None):
            acc = accumulator(acc,U0,U,dt,k,tsteps,Dcoeff,omega,r2decay)

        # Update
        U0.assign(U)

    return (U, acc)

# ---------------------------------------------------------------------------- #
# Solve the Bloch-Torrey equation with linear finite elements, time-stepping
# using the backward euler method and the TRBDF2 method with α = 2-√2
# ---------------------------------------------------------------------------- #
def run_bt():
    # isvesselunionlist = [True, False]
    # isvesselunionlist = [True]
    isvesselunionlist = [False]

    # Nlist, nslicelist = [10,25,50,100], [8,16,32,32]
    # Nlist, nslicelist = [10,25,50], [8,16,32]
    Nlist, nslicelist = [10], [8]
    # Nlist, nslicelist = [200], [64]

    # Number of simulations to do, halving time step each time.
    # Minimum time step dt_min = dt0/2^(Num-1):
    #   9 -> dt_min = 8e-3/2**8 = 3.125e-5
    #   8 -> dt_min = 8e-3/2**7 = 6.25e-5
    #   7 -> dt_min = 8e-3/2**6 = 0.000125
    dt0 = 8.0e-3 # [s]
    NumBE = 1
    NumTR = 1

    # parent_foldname = 'bt/results'
    parent_foldname = 'bt/tmp'
    vesselradius = 250.0
    Dcoeff = 3037.0

    for isvesselunion in isvesselunionlist:
        for N, nslice in zip(Nlist,nslicelist):

            print("\n", "isvesselunion = ", isvesselunion, ", N = ", N, ", nslice = ", nslice, "\n")

            Geomargs = {'N':N, 'nslice':nslice, 'vesselrad':vesselradius, 'isvesselunion':isvesselunion}
            Gammaargs = {'a':vesselradius, 'force_outer':not isvesselunion}

            V, elem, mesh, mesh_str = get_bt_geom(**Geomargs)
            omega, r2decay = create_bt_gamma(V, mesh, **Gammaargs)

            sub_foldname = 'union' if isvesselunion else 'hollow';
            results_foldname = parent_foldname + '/' + sub_foldname + '/' + mesh_str

            # ---------------------------------------------------------------- #
            # Backward Euler Method
            # ---------------------------------------------------------------- #
            U0 = Function(V, name='InitMag')
            U0.assign(Constant((0.0, 1.0))) # π/2-pulse
            stepper = 'be'
            dt = dt0
            for _ in range(NumBE):
                BEfoldname = results_foldname + '/' + stepper + '/dt_' + str(dt).replace('.','p')
                BEargs = {'dt':dt, 'stepper':stepper, 'savesignal':True, 'foldname':BEfoldname}

                print("\n", "BE: dt = ", dt, "\n")
                bt_solver(U0, V, mesh, Dcoeff, omega, r2decay, **BEargs)

                dt = 0.5*dt

            # ---------------------------------------------------------------- #
            # TRBDF2 (α = 2-√2) Method
            # ---------------------------------------------------------------- #
            U0 = Function(V, name='InitMag')
            U0.assign(Constant((0.0, 1.0))) # π/2-pulse
            stepper = 'trbdf2'
            dt = dt0
            for _ in range(NumTR):
                TRfoldname = results_foldname + '/' + stepper + '/dt_' + str(dt).replace('.','p')
                TRargs = {'dt':dt, 'stepper':stepper, 'savesignal':True, 'foldname':TRfoldname}

                print("\n", "TRBDF2: dt = ", dt, "\n")
                bt_solver(U0, V, mesh, Dcoeff, omega, r2decay, **TRargs)

                dt = 0.5*dt

    return

def run_adaptive_bt():
    # isvesselunionlist = [True]
    # isvesselunionlist = [False]
    isvesselunionlist = [True, False]

    # Nlist, nslicelist = [10,25,50,100], [8,16,32,32]
    # Nlist, nslicelist = [10,25,50], [8,16,32]
    # Nlist, nslicelist = [200], [64]
    # Nlist, nslicelist = [50], [32]
    # Nlist, nslicelist = [25], [16]
    # Nlist, nslicelist = [10], [8]
    # Nlist, nslicelist = [10], [32]
    Nlist, nslicelist = [10], [64]
    # Nlist, nslicelist = [10, 10], [32, 64]

    # High accuracy solution to compare to:
    # N_exlist, nslice_exlist = [100], [8]
    # N_exlist, nslice_exlist = [100], [32]
    # N_exlist, nslice_exlist = [50], [32]
    # N_exlist, nslice_exlist = [100], [32]
    N_exlist, nslice_exlist = [200], [64]
    # N_exlist, nslice_exlist = [100, 200], [32, 64]

    # stepper_list = ['be']
    stepper_list = ['be', 'trbdf2']

    # dt_list = [1.0e-3] # [s]
    # dt_list = [0.5e-3, 0.25e-3] # [s]
    dt_list = [4.0e-3, 2.0e-3, 1.0e-3, 0.5e-3]

    vesselradius_list = [250.0, 125.0] # [μm]
    Dcoeff_list = [3037.0] # [mm²/s]
    T_list = [40.0e-3] # [s]

    parent_foldname = 'bt/adapt/results'
    # parent_foldname = 'bt/adapt/tmp'
    max_mesh_refinements = 4
    refine_percentile = 70.0 # cutoff percentile above which cell is refined
    refine_thresh = 1e-4 * (3000.0)**3

    compute_high_accuracy = True # High accuracy solution to compare to
    skip_last_dual = True # skip last dual iteration (won't print last |e⋅ψ|/S)
    exactprnt = True
    forwprnt = True
    dualprnt = True

    save_forw_signal = True
    save_dual_signal = True
    save_highprec_signal = True
    save_forw_mag = False
    save_dual_mag = False
    save_highprec_mag = False
    save_eta_y = False

    def print_u_detailed(U,S0=1.0):
        Sx, Sy, S = S_signal(U)
        print("\n")
        print('[Sx/S, Sy/S] = [', Sx/S0, ', ', Sy/S0, ']')
        print("Sx = ", Sx)
        print("Sy = ", Sy)
        print("S  = ", S)
        print("|Sx_err|/Sx = ", abs(Sx-Sx_ex)/Sx_ex)
        print("|Sy_err|/Sy = ", abs(Sy-Sy_ex)/Sy_ex)
        print(" |S_err|/S  = ", abs(S-S_ex)/S_ex)
        print("\n")
        return

    param_lists = (dt_list, T_list, stepper_list, isvesselunionlist, vesselradius_list, Dcoeff_list)

    for params in itertools.product(*param_lists):

        # Unpack parameters
        dt, T, stepper, isvesselunion, vesselradius, Dcoeff = params
        union_or_hollow = 'union' if isvesselunion else 'hollow'
        sub_foldname = union_or_hollow + '/' + stepper;

        for N, nslice, N_ex, nslice_ex in zip(Nlist,nslicelist,N_exlist,nslice_exlist):

            mesh_str = get_mesh_str(N,nslice,vesselradius)

            if compute_high_accuracy:
                # ---------------------------------------------------------------- #
                # Compute high accuracy solution on fine grid
                # ---------------------------------------------------------------- #
                print("\n", "High Precision Solution (", stepper, "): dt = ", dt, "\n")

                Geomargs = {'N':N_ex, 'nslice':nslice_ex, 'vesselrad':vesselradius, 'isvesselunion':isvesselunion}
                V, elem, mesh, mesh_str_ex = get_bt_geom(**Geomargs)

                results_foldname_ex = parent_foldname + '/' + sub_foldname + '/' + mesh_str
                foldname_ex = results_foldname_ex + '/Forw/dt_' + str(dt).replace('.','p') + \
                              '/exact/' + mesh_str_ex

                Gammaargs = {'a':vesselradius, 'force_outer':not isvesselunion}
                omega, r2decay = create_bt_gamma(V, mesh, **Gammaargs)

                U0 = Function(V, name='InitMag')
                U0.assign(Constant((0.0, 1.0))) # π/2-pulse
                Sx0, Sy0, S0 = S_signal(U0)

                ExArgs = {'dt':dt, 'T':T, 'stepper':stepper, 'prnt':exactprnt,
                          'savesignal':save_highprec_signal, 'savemag':save_highprec_mag,
                          'foldname':foldname_ex}
                U_ex, _ = bt_solver(U0, V, mesh, Dcoeff, omega, r2decay, **ExArgs)
                Sx_ex, Sy_ex, S_ex = S_signal(U_ex)
                print_u_detailed(U_ex,S0=S0)

            print("\n", "isvesselunion = ", isvesselunion, ", N = ", N, ", nslice = ", nslice, "\n")

            Geomargs = {'N':N, 'nslice':nslice, 'vesselrad':vesselradius, 'isvesselunion':isvesselunion}
            V, elem, mesh, _ = get_bt_geom(**Geomargs)

            # Generate initial geometry
            for k in range(max_mesh_refinements+1): # first run is unrefined

                # Generate omega and r2decay maps
                Gammaargs = {'a':vesselradius, 'force_outer':not isvesselunion}
                omega, r2decay = create_bt_gamma(V, mesh, **Gammaargs)

                results_foldname = parent_foldname + '/' + sub_foldname + '/' + mesh_str
                # ------------------------------------------------------------ #
                # Solve forward problem using backward Euler method
                # ------------------------------------------------------------ #
                print("\n", "Forward problem: dt = ", dt, "\n")

                U = Function(V, name='InitMag')
                U.assign(Constant((0.0, 1.0))) # π/2-pulse
                Sx0, Sy0, S0 = S_signal(U)

                foldname = results_foldname + '/Forw/dt_' + str(dt).replace('.','p') + '/iter' + str(k)
                ForwArgs = {'isdual':False, 'stepper':stepper, 'dt':dt, 'T':T,
                            'accumulator':accum_max_vecnormdiff, 'prnt':forwprnt,
                            'savesignal':save_forw_signal, 'savemag':save_forw_mag, 'foldname':foldname}
                U, Eu = bt_solver(U, V, mesh, Dcoeff, omega, r2decay, **ForwArgs)
                Sx, Sy, S = S_signal(U)

                # ------------------------------------------------------------ #
                # Compare with high accuracy solution
                # ------------------------------------------------------------ #
                if compute_high_accuracy:
                    print_u_detailed(U,S0=S0)
                else:
                    print('[Sx/S, Sy/S] = [', Sx/S0, ', ', Sy/S0, ']')

                if skip_last_dual and (k == max_mesh_refinements):
                    break

                # ------------------------------------------------------------ #
                # Solve Dual problem using backward Euler method
                # ------------------------------------------------------------ #
                print("\n", "Dual problem: dt = ", dt, "\n")

                Z = TestFunction(V)
                w, z = split(Z) # test function -> components

                # Assemble the inital vector psi for the dual problem. psi is a
                # vector containing quadrature weights for computing Sy, i.e.:
                #   U⋅ψ = ∫vdx
                # where U = (u,v)
                psi = Function(V, name='InitDualMag')
                psi.vector()[:] = assemble(z*dx)
                # psi_norm = np.linalg.norm(psi.vector())
                # psi.vector()[:] /= psi_norm # normalize psi

                foldname = results_foldname + '/Dual/dt_' + str(dt).replace('.','p') + '/iter' + str(k)
                DualArgs = {'isdual':True, 'stepper':stepper, 'dt':dt, 'T':T,
                            'accumulator':accum_dual, 'prnt':dualprnt,
                            'savesignal':save_dual_signal, 'savemag':save_dual_mag, 'foldname':foldname}

                phi = get_dual_init(psi, V)
                phi, acc = bt_solver(phi, V, mesh, Dcoeff, omega, r2decay, **DualArgs)
                int_phi_vec_sq, int_phi_norm = acc

                Sy_err0 = Eu * int_phi_norm
                Sy_err1 = Eu * sqrt(T) * np.sqrt(np.sum(int_phi_vec_sq.vector().get_local()))

                SY = Sy_ex if compute_high_accuracy else Sy
                print("\n", "|e⋅ψ|/|Sy| = |Sy_err|/|Sy| < ", Sy_err0/SY, "\n")
                print("\n", "|e⋅ψ|/|Sy| = |Sy_err|/|Sy| < ", Sy_err1/SY, "\n")

                if compute_high_accuracy:
                    print("\n", "|error bound|/|true error| = ", Sy_err1/abs(Sy-SY), "\n")

                # Create indicators
                eta_x, eta_y = int_phi_vec_sq.split()
                DG = FunctionSpace(mesh, 'DG', 0)
                Eta_y = interpolate(eta_y, DG)
                Eta_y.vector()[:] = Eu * sqrt(T) * np.sqrt(Eta_y.vector())

                # Create VTK files for output
                if save_eta_y:
                    vtkfile_err_y = File(foldname + '/err/' + 'Eta_y.pvd')
                    vtkfile_err_y << Eta_y

                if k == max_mesh_refinements:
                    break

                # Make cell function
                cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
                cell_markers.set_all(False)

                Eta_y_thresh = np.percentile(Eta_y.vector(),refine_percentile)
                Eta_y_thresh = np.maximum(Eta_y_thresh, refine_thresh)

                is_any_true = False
                for cell_idx in xrange(mesh.num_cells()):
                    cell = Cell(mesh, cell_idx)
                    if Eta_y.vector()[cell_idx] >= Eta_y_thresh:
                        cell_markers[cell] = True
                        is_any_true = True

                # Refine mesh
                if is_any_true:
                    mesh = refine(mesh, cell_markers)
                    V = FunctionSpace(mesh, elem)
                else:
                    break

    return

if __name__ == "__main__":
    # run_bt()
    run_adaptive_bt()
