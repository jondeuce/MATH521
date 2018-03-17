% hw1.m
clear all; close all; clc;

% sample function
u = @(x1,x2) sin(2.*pi.*x1).*cos(6.*pi.*x2);

% mesh the rectange [0,1] x [2,3] with 20 / 60 subintervals in x1- / x2-direction, respectively
msh = meshRectangle([0,1,2,3],[20,60]);

% evaluate u on msh and draw a surface plot
figure; surf(msh.X1,msh.X2,u(msh.X1,msh.X2));

% axis labels
xlabel('{\itx}_1')
ylabel('{\itx}_2')
zlabel('{\itu}({\itx}_1,{\itx}_2)')