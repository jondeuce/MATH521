# coding=utf-8
"""
Old solvers
"""

def bt_bwdeuler(U0, V, mesh, omega, r2decay, isdual = False,
                t0 = 0.0, T = 40.0e-3, dt = 1.0e-3, Dcoeff = 3037.0,
                prnt = True, savesignal = False, savemag = False,
                foldname = 'be/tmp'):
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
    A = bt_bilinear(W,Z,Dcoeff,r2decay,omega,isdual=isdual)
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
    U.assign(U0)

    # Compute initial signal
    Sx0, Sy0, S0 = S_signal(U0)

    if prnt:
        print('Step = ', 0, '/', tsteps , 'Time =', t0)
        print_u(U0, S0=S0)

    if savesignal:
        # check if folder exists
        if not os.path.exists(foldname):
            os.makedirs(foldname)

        # write signal
        signal = csv.writer(open(foldname + '/' + 'signal.csv', 'w'))
        signal.writerow([t] + [Sx0] + [Sy0] + [S0] + [timer()-funcstart])

    if savemag:
        # Create VTK files for visualization output and save initial state
        vtkfile_u = File(foldname + '/' + 'u.pvd')
        vtkfile_v = File(foldname + '/' + 'v.pvd')
        u, v = U0.split()
        vtkfile_u << (u, t0)
        vtkfile_v << (v, t0)

    # Time-stepping
    for k in range(tsteps):
        # start loop time
        loopstart = timer()

        # Current time
        t = t0 + (k+1)*dt

        # Assemble the right hand (backward euler step; U = U0 here)
        #   dU/dt = -A*U => (M + A*dt)U = M*U0
        L = M_bilinear(U0,Z)
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

        if savesignal:
            signal.writerow([t] + [Sx] + [Sy] + [S] + [timer()-funcstart])

        if savemag:
            u, v = U.split()
            vtkfile_u << (u, t)
            vtkfile_v << (v, t)

        U0.assign(U)

    return U

def bt_trbdf2(U0, V, mesh, omega, r2decay, isdual = False,
              t0 = 0.0, T = 40.0e-3, dt = 1.0e-3, Dcoeff = 3037.0,
              prnt = True, savesignal = False, savemag = False,
              foldname = 'trbdf2/tmp'):
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
    A = bt_bilinear(W,Z,Dcoeff,r2decay,omega,isdual=isdual)
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
        vtkfile_u = File(foldname + '/' + 'u.pvd')
        vtkfile_v = File(foldname + '/' + 'v.pvd')
        u0, v0 = U0.split()
        vtkfile_u << (u0, t0)
        vtkfile_v << (v0, t0)

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
        if savesignal:
            signal.writerow([t] + [Sx] + [Sy] + [S] + [timer()-funcstart])

        if savemag:
            vtkfile_u << (u, t)
            vtkfile_v << (v, t)

        # Update
        U0.assign(U)

    return U
