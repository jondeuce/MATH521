function [A,b] = discretisePoisson(f,g,msh)
%DISCRETISEPOISSON Assembles the system matrix A and the right hand side
%vector b for Poisson's equation with Dirichlet boundary conditions.
%   Input:
%       f: a function handle for the inhomogeneity in Poisson's equation
%       g: a function handle for the boundary values
%       msh: a mesh structure
%   Output:
%       A: a sparse array
%       b: a column vector

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% x1 derivatives
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 2nd difference operator in 1D
L = spdiags(kron([-1 2 -1],ones(msh.N(1)-1,1)),-1:1,msh.N(1)-1,msh.N(1)-1);

% identity matrix & vectors [1;0;...;0], [0;...;0;1]
E = speye(msh.N(2)-1); e1 = full(E(:,[1 end]));

% -d²/dx1²
A = kron(E,L)/msh.h(1)^2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% x2 derivatives
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 2nd difference operator in 1D
L = spdiags(kron([-1 2 -1],ones(msh.N(2)-1,1)),-1:1,msh.N(2)-1,msh.N(2)-1);

% identity matrix & vectors [1;0;...;0], [0;...;0;1]
E = speye(msh.N(1)-1); e2 = full(E(:,[1 end]));

% -d²/dx2²
A = A + kron(L,E)/msh.h(2)^2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% right hand side
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% boundary data
gN = g(msh.X1(end,2:end-1),msh.X2(end,2:end-1))';
gE = g(msh.X1(2:end-1,end),msh.X2(2:end-1,end));
gS = g(msh.X1(1,2:end-1),msh.X2(1,2:end-1))';
gW = g(msh.X1(2:end-1,1),msh.X2(2:end-1,1));

% discrete source term
b = msh2vec(f(msh.X1(2:end-1,2:end-1),msh.X2(2:end-1,2:end-1)),msh) + ...
    kron(gW,e2(:,1))/msh.h(1)^2 + kron(gE,e2(:,2))/msh.h(1)^2 + ...
    kron(e1(:,1),gS)/msh.h(2)^2 + kron(e1(:,2),gN)/msh.h(2)^2;