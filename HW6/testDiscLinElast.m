%% create forcing function solutions
syms x1 x2 real
syms a c positive
PI = sym('pi');
% u = sin(2*PI*x1)*sin(2*PI*x2);
u = sin(x1+x2)^2;

F = matlabFunction(simplify( -c*(diff(u,x1,2)+diff(u,x2,2)) + a*u ));
U = matlabFunction(u);

%% set params
c = 1;
a = 1;
% f = @(x1,x2) ones(size(x1)); %bsxfun(@times, sin(x1), cos(x2)) + 3.4;
% g = @(x1,x2) zeros(size(x1)); %bsxfun(@times, x1, x2) + 1.7;
f = @(x1,x2) F(a,c,x1,x2);
g = U;
GammaD = @(x1,x2) true(size(x1));

% msh = load('video10.mat');
% msh = load('kiwi.mat');
xb = [-1,1]; yb = [-2,2]; Nx = 300; Ny = 500;
msh = rectMeshFEM( xb, yb, Nx, Ny );

%% get discrete linear elasticity operators
[A,b,uD,Kbar,Mbar,Pf,PD] = discretiseLinearElasticity(c,a,f,g,GammaD,msh);

%% solve system
uN = A\b;
uh = Pf'*uN + PD'*uD;

%% norm of difference with true solution
u_ex = g(msh.P(1,:),msh.P(2,:)).';
fprintf('\n');
fprintf(' norm(u):\t%.4e\n', norm(u_ex));
fprintf('max(|u|):\t%.4e\n', maxabs(u_ex));
fprintf(' rel-err:\t%.4e\n', norm(u_ex-uh)/norm(u_ex));
fprintf(' max-err:\t%.4e\n', maxabs(u_ex-uh));

%% plot solution
% pdeplot(msh.P,msh.E,msh.T,'xydata',uh); title('solution');
% pdeplot(msh.P,msh.E,msh.T,'xydata',U-uh); title('error');
