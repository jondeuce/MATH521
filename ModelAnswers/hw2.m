% hw2.m
close all; clc;

% sample function
u = @(x1,x2) x1.*x2.*sin(2.*pi.*x1).*cos(6.*pi.*x2);

% mesh the rectange [0,1] x [2,3] with 20 / 60 subintervals in x1- / x2-direction, respectively
msh = meshRectangle([0,1,2,3],[20,60]);

% test msh2vec and vec2msh
U =  u(msh.X1(2:end-1,2:end-1),msh.X2(2:end-1,2:end-1));
V = vec2msh(msh2vec(U,msh),msh);
fprintf('||U - V|| = %g\n',norm(U-V));
