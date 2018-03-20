function [ A, b ] = discretisePoisson( f, g, msh )
%DISCRETISEPOISSON Assembles the linear system for the Poisson-Dirichlet 
% problem for the forcing function f, boundary value function g, and mesh
% msh. A is a sparse matrix, b a vector, and A\b produces solution for the
% interior grid points.
% 
% The Poisson-Dirichlet problem is defined as:
%   -Lap(u) = f  inside 'msh'
%        u  = g  on the boundary of 'msh'
% 
% INPUTS 
%   f:   2-argument function handle for the forcing function f(x1,x2)
%   g:   2-argument function handle for the boundary function g(x1,x2)
%   msh: Struct containing grid information (output of meshRectangle)
% 
% OUTPUTS
%   A:   A sparse (msh.N(1)-1)*(msh.N(2)-1) x (msh.N(1)-1)*(msh.N(2)-1)
%        array representing the discretized Poisson-Dirichlet system
%   b:   A (msh.N(1)-1)*(msh.N(2)-1) x 1 vector representing the forcing 
%        terms and boundary terms

% Convenience variables
[Nx,Ny,hx,hy] = deal(msh.N(1), msh.N(2), msh.h(1), msh.h(2));
[ex,ey] = deal(ones(Nx-1,1), ones(Ny-1,1));
[Ix,Iy] = deal(speye(Nx-1), speye(Ny-1));

% Constructing b: initialize with forcing term values
b = f(msh.X1(2:end-1,2:end-1), msh.X2(2:end-1,2:end-1));

% Constructing b: add appropriate boundary terms and reshape to a vector
b(1,:)   = b(1,:)   + (1/hy^2) * g(msh.X1(1,2:end-1),   msh.X2(1,2:end-1));
b(end,:) = b(end,:) + (1/hy^2) * g(msh.X1(end,2:end-1), msh.X2(end,2:end-1));
b(:,1)   = b(:,1)   + (1/hx^2) * g(msh.X1(2:end-1,1),   msh.X2(2:end-1,1));
b(:,end) = b(:,end) + (1/hx^2) * g(msh.X1(2:end-1,end), msh.X2(2:end-1,end));
b        = msh2vec(b,msh);

% Constructing A: 2nd difference block matrices for x- and y-direction
Lx = (1/hx^2) * spdiags( [-ex, 2*ex, -ex], [-1 0 1], Nx-1, Nx-1 );
Ly = (1/hy^2) * spdiags( [-ey, 2*ey, -ey], [-1 0 1], Ny-1, Ny-1 );

% Constructing A: Assemble using the kronecker product with appropriate
% identity matrices
A  = kron(Iy,Lx) + kron(Ly,Ix);

end

