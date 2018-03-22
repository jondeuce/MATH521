%% clear workspace
clear all
close all force

%% set params
c = 0.1;
a = 1;
f = @(x1,x2) ones(size(x1)); % uniform forcing
g = @(x1,x2) zeros(size(x1)); % fixed boundaries
GammaD = @(x1,x2) true(size(x1));
% GammaD = @(x1,x2) false(size(x1));

%% choose a mesh
% fname = 'kiwi.mat';
fname = 'maple.mat';
% fname = 'pi.mat';
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

%% undistorted plots of mesh, force function f, and solution uh
figure, subplot(3,1,1);
pdeplot(msh.P,msh.E,msh.T);
axis equal
xlim([min(msh.P(1,:)), max(msh.P(1,:))]);
ylim([min(msh.P(2,:)), max(msh.P(2,:))]);
title(sprintf('FEM Mesh: %s',fname))

subplot(3,1,2);
spy(Mbar);
title('Mass Matrix Sparsity Pattern');

subplot(3,1,3);
spy(Kbar);
title('Stiffness Matrix Sparsity Pattern');

%% undistorted plots of force function f and solution uh
fh = f(msh.P(1,:),msh.P(2,:)).';
figure, pdeplot(msh.P,msh.E,msh.T,'xydata',fh);
axis equal, title('Forcing Function $f$');

figure, pdeplot(msh.P,msh.E,msh.T,'xydata',uh);
axis equal
NUM2STR = @(x) strrep(sprintf('%.2e',x),'e-','e\!-\!');
titlestr = ['$u^h$: ', ...
    '$||u^h||_{L^2} = ', NUM2STR(L2norm), '$, ' ...
    '$||u^h||_{H^1} = ', NUM2STR(H1norm), '$, ' ...
    '$||u^h||_{B} = ', NUM2STR(Bnorm), '$' ];
title(titlestr);
