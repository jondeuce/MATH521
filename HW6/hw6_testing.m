ccc

%% create symbolic test solution
syms x1 x2 real
syms a c positive
PI = sym('pi');
u = sin(2*PI*x1)*sin(2*PI*x2);

%% create function handles
F = matlabFunction(simplify( -c*(diff(u,x1,2)+diff(u,x2,2)) + a*u ));
U = matlabFunction(u);

%% set params
c = 1;
a = 1;
% f = @(x1,x2) ones(size(x1)); % uniform forcing
% g = @(x1,x2) zeros(size(x1)); % zero boundaries
f = @(x1,x2) F(a,c,x1,x2);
g = U;
GammaD = @(x1,x2) true(size(x1));
% GammaD = @(x1,x2) false(size(x1));

%% choose a mesh
% fname = 'kiwi.mat';
% fname = 'maple.mat';
fname = 'pi.mat';
% fname = 'ubc.mat';
% fname = 'video10.mat';
msh = load(fname);

%% get discrete linear elasticity operators
[A,b,uD,Kbar,Mbar,Pf,PD] = discretiseLinearElasticity(c,a,f,g,GammaD,msh);

%% solve system
uN = A\b;
uh = Pf'*uN + PD'*uD;

%% calculate various norms
L2normsq = uh'*Mbar*uh;
L2gradsq = uh'*Kbar*uh;

L2norm = sqrt(L2normsq);
H1norm = sqrt(L2normsq + L2gradsq);
Bnorm = sqrt(c*L2gradsq + a*L2normsq);

%% norm of difference with true solution
u_ex = g(msh.P(1,:),msh.P(2,:)).';
fprintf('\n');
fprintf(' norm(u):\t%.4e\n', norm(u_ex));
fprintf('max(|u|):\t%.4e\n', maxabs(u_ex));
fprintf(' rel-err:\t%.4e\n', norm(u_ex-uh)/norm(u_ex));
fprintf(' max-err:\t%.4e\n', maxabs(u_ex-uh));

%% undistorted plots of mesh, force function f, and solution uh
figure, subplot(1,3,1);
pdeplot(msh.P,msh.E,msh.T);
axis equal
xlim([min(msh.P(1,:)), max(msh.P(1,:))]);
ylim([min(msh.P(2,:)), max(msh.P(2,:))]);
title(sprintf('FEM Mesh: %s',fname))

subplot(1,3,2);
spy(Mbar);
title('Mass Matrix Sparsity Pattern');

subplot(1,3,3);
spy(Kbar);
title('Stiffness Matrix Sparsity Pattern');

%% undistorted plots of force function f and solution uh
fh = f(msh.P(1,:),msh.P(2,:)).';
figure, pdeplot(msh.P,msh.E,msh.T,'xydata',fh);
axis equal, title('Forcing Function $f$');

figure, pdeplot(msh.P,msh.E,msh.T,'xydata',uh);
axis equal, title('Solution Function $u^h$');
