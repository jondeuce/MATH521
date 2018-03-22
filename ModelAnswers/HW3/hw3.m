% hw3.m
close all; clc;

% data
f = @(x1,x2) 40*pi^2*sin(2*pi*x1).*cos(6*pi*x2);
g = @(x1,x2) sin(2*pi*x1).*cos(6*pi*x2);

% mesh the rectange [0,1] x [2,3] with 20 / 60 subintervals in x1- / x2-direction, respectively
msh = meshRectangle([0,1,2,3],[20,60]);

% Dirichlet boundary conditions
U = g(msh.X1,msh.X2);

% solve the Poisson-Dirichlet problem
[A,b] = discretisePoisson(f,g,msh);
U(2:end-1,2:end-1) = vec2msh(A\b,msh); % finite-difference solution

% draw a surface plot
figure; surf(msh.X1,msh.X2,U);
xlabel('{\itx}_1')
ylabel('{\itx}_2')
zlabel('{\itu^h}({\itx}_1,{\itx}_2)')

% sparsity pattern of A
figure; spy(A);
