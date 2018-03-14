%% hw3.m
clear all; close all; clc;

%% forcing function and boundary function
f = @(x1,x2) (40*pi^2) * sin(2.*pi.*x1).*cos(6.*pi.*x2);
g = @(x1,x2) sin(2.*pi.*x1).*cos(6.*pi.*x2);

%% Construct [0,1]x[2,3] mesh with 20/60 subintervals in x1-/x2-direction
msh = meshRectangle([0,1,2,3], [20,60]);

%% Get discretized Poisson-Dirichlet problem
[A,b] = discretisePoisson(f,g,msh);

%% Preallocate u and set the boundary values to g
u = zeros(msh.N(2)+1,msh.N(1)+1);
u(1,:)         = g(msh.X1(1,:),         msh.X2(1,:));
u(end,:)       = g(msh.X1(end,:),       msh.X2(end,:));
u(2:end-1,1)   = g(msh.X1(2:end-1,1),   msh.X2(2:end-1,1));
u(2:end-1,end) = g(msh.X1(2:end-1,end), msh.X2(2:end-1,end));

%% Solve the problem for the interior points and assign to the interior of u
u(2:end-1,2:end-1) = vec2msh(A\b, msh); % interior points

%% Plotting
% Convenience variables
textargs = {'fontsize',24,'interpreter','latex'};
markersize = 10;

% Draw a surface plot of solution u
figure('color','w'), grid on
surf(msh.X1, msh.X2, u);

% axis labels
xlabel('$x_1$', textargs{:});
ylabel('$x_2$', textargs{:});
zlabel('$u(x_1,x_2)$', textargs{:});
title('Poisson-Dirichlet Solution (with boundary points)', textargs{:});

% save figure to pdf
export_fig hw3_PoissonSolution -transparent -pdf

% Plot structure of A matrix
figure('color','w'), grid on

subplot(1,2,1);
spy(A);
title('Matrix A Full structure',textargs{:});

subplot(1,2,2);
Nmax = max(msh.N);
spy(A(1:Nmax,1:Nmax));
title(sprintf('A(1:%d,1:%d) Structure',Nmax,Nmax),textargs{:});

% save figure to pdf
export_fig hw3_MatrixStructure -transparent -pdf
