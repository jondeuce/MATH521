% hw1.m
clear all; close all; clc;

% sample function
u = @(x1,x2) sin(2.*pi.*x1).*cos(6.*pi.*x2);

% mesh the rectange [0,1] x [2,3] with 20 / 60 subintervals in 
% x1-/x2-direction, respectively
msh = meshRectangle([0,1,2,3], [20,60]);

% evaluate u on msh and draw a surface plot
figure, grid on
surf(msh.X1, msh.X2, u(msh.X1,msh.X2));

% axis labels
textargs = {'fontsize',24,'interpreter','latex'};
xlabel('$x_1$', textargs{:});
ylabel('$x_2$', textargs{:});
zlabel('$\sin(2\pi x_1)\cos(6\pi x_2)$', textargs{:});

% set background to white
set(gcf,'color','w');

% save figure to pdf
export_fig hw1_figure -transparent -pdf