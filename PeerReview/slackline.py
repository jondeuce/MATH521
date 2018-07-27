# coding=utf-8
from __future__ import print_function
from fenics import *
from mshr import *
import time
import numpy as np

# Parameters
L_gap = 50.
L_slackline = L_gap *0.98
elasticity = 12. # % at 10 kN
mu_webbing = 0.052 # kg/m
mass_slackliner = 55. # kg
width_slackliner = 0.6
position_slackliner = 0.5 * (L_gap - width_slackliner)

g = 9.81
non_phy=1e-12 # needed for convergence, to tune

k = 1./(elasticity*1e-6) # N

# Time interval
t0 = 0.
T = 4.0
t = t0

mu_slackliner = Expression('mass_slackliner/width_slackliner*(x[0]>=position_slackliner)*(x[0]<=position_slackliner+width_slackliner)', degree=0,mass_slackliner=mass_slackliner,position_slackliner=position_slackliner,width_slackliner=width_slackliner)
M = Expression(('0','g*mu_slackliner*sin(2*pi*10*t)/10*(t<=0.1)','0'), degree=0, g=g,mu_slackliner=mu_slackliner,t=t)
mu_tot = mu_webbing*L_gap/L_slackline+mu_slackliner

# Create mesh and define function space
N = int(40*L_gap)
hmin=L_gap/N
mesh = IntervalMesh(N, 0, L_gap)
for j in range(2):
    cell_markers = CellFunction("bool", mesh)
    cell_markers.set_all(False)
    for cell in cells(mesh):
        if cell.distance(Point(0,0,0))>=position_slackliner-1. and cell.distance(Point(0,0,0))<=position_slackliner+width_slackliner+1.:
            cell_markers[cell]=True
    mesh = refine(mesh, cell_markers)
    hmin=hmin/2
h=CellSize(mesh)
V = VectorFunctionSpace(mesh, 'P', 2, 3) # 2 is degree

# Define boundary condition
d_D = Constant((0, 0, 0))
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, d_D, boundary)

# Define steady variational problem (ODE)
d0 = TrialFunction(V)
w = TestFunction(V)
e_z = Constant((0, 0, 1))
e_x = Constant((1, 0, 0))
J = sqrt(dot(e_x + d0.dx(0),e_x + d0.dx(0)))
el = L_gap/L_slackline*J-1
el = (el+non_phy+abs(el-non_phy))/2 # max(non_phy,el) rewritten...
wel = w*el
F=dot(-g*J*mu_tot/k*e_z+(J*el*(1/J).dx(0)+el.dx(0))*(e_x+d0.dx(0)),w)*dx - dot(d0.dx(0),wel.dx(0))*dx

# Compute steady solution
d0=Function(V)
F=action(F,d0)
Jacobian=derivative(F,d0)
problem = NonlinearVariationalProblem(F, d0, bc, Jacobian)
solver = NonlinearVariationalSolver(problem) # tune solver ?
solver.solve()

# Set initial data
d = Function(V)
v = Function(V)
d.assign(d0)
v0 = interpolate(Constant((0,0,0)), V)
v.assign(v0)
displacement = File('slackline/dispacement.pvd')
wiggle = File('slackline/wiggles.pvd')
d.rename("displacement", d.name())
displacement << (d,t)
(disp_x,disp_y,disp_z) = d.split()
disp_y.rename("wiggle", disp_y.name())
wiggle << (disp_y,t)
##velocity = File('slackline/v.pvd')
##v.rename("velocity", v.name())
##velocity << (v,t)
##
##mistake = File('slackline/mistake.pvd')
##Mf=interpolate(M,V)
##Mf.rename("mistake",Mf.name())
##mistake << (Mf,t)

# Find appropriate time-step
CFL = 1.
J = sqrt(dot(e_x + d0.dx(0),e_x + d0.dx(0)))
el = L_gap/L_slackline*J-1
el = (el+non_phy+abs(el-non_phy))/2 # max(non_phy,el) rewritten...
DG = TestFunction(FunctionSpace(mesh,'DG',0))
Tension = k*el*DG/h*dx
Tension = max(np.abs(assemble(Tension)))
u = sqrt(Tension/mu_webbing)
dtCFL = CFL*hmin/u
dt = 0.01
tsteps = int((T-t0)/dt)
print('hmin = ',hmin,'u = ',u,'dt = ', dt,'tension',Tension,'dt_CFL',dtCFL)
saved=401 # to reduce folder size
walk=0. # walkability indicator

# Time stepping
for j in range(tsteps):

    # Current time
    t = t0 + (j+1)*dt
    print('Step = ', j+1, '/', tsteps , 'Time =', t)

    tm = t0 + (j+1.5)*dt
    M.t = tm

    # First 1st order in time problem
    dp = TrialFunction(V)
    dm = (d+dp)/2.
    J = sqrt(dot(e_x + dm.dx(0),e_x + dm.dx(0)))
    el = L_gap/L_slackline*J-1
    el = (el+non_phy+abs(el-non_phy))/2 # max(non_phy,el) rewritten...
    wel = w*el
    F = (2*mu_tot*J/(dt*k)*dot((dp-d)/dt-v,w) - dot(-g*J*mu_tot/k*e_z+J/k*M+(J*el*(1/J).dx(0)+el.dx(0))*(e_x+dm.dx(0)),w) + dot(dm.dx(0),wel.dx(0)))*dx
    dp = Function(V)
    F = action(F, dp)
    Jacobian = derivative(F,dp)
    dp.assign(d)
    problem = NonlinearVariationalProblem(F, dp, bc, Jacobian)
    solver = NonlinearVariationalSolver(problem) # tune solver ?
    #prm = solver.parameters
    #prm["newton_solver"]["absolute_tolerance"] = 1E-16
    #prm["newton_solver"]["relative_tolerance"] = 1E-16
    #prm["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = True
    #info(prm, True)
    solver.solve()

    dm = (d+dp)/2.
    # Second 1st order in time problem LINEAR !!!
    vp = TrialFunction(V)
    J = sqrt(dot(e_x + dm.dx(0),e_x + dm.dx(0)))
    el = L_gap/L_slackline*J-1
    el = (el+non_phy+abs(el-non_phy))/2 # max(non_phy,el) rewritten...
    wel = w*el
    F = (mu_tot*J/(dt*k)*dot(vp-v,w) - dot(-g*J*mu_tot/k*e_z+J/k*M+(J*el*(1/J).dx(0)+el.dx(0))*(e_x+dm.dx(0)),w) + dot(dm.dx(0),wel.dx(0)))*dx
    B=lhs(F)
    L=rhs(F)
    vp = Function(V)
    solve(B == L, vp, bc)
    (disp_x,disp_y,disp_z)=dp.split()
    walkp=max(np.abs(assemble(disp_y*DG/h*dx)))
    walk=max(walk,walkp)
    
    # Write data to file when requested
    if np.mod(j+1,saved)==0:
        dp.rename("displacement", dp.name())
        displacement << (dp, t)
        disp_y.rename("wiggle", disp_y.name())
        wiggle << (disp_y,t)
##        vp.rename("velocity", vp.name())
##        velocity << (vp,t)
##        Mf=interpolate(M,V)
##        Mf.rename("mistake",Mf.name())
##        mistake << (Mf,t)

    # Update
    d.assign(dp)
    v.assign(vp)
print('Tension = ', Tension/1000, 'Walkability indicator (maximal lateral displacement) = ', walk*1000)
