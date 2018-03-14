%% test.m
ccc

%% syms variables
x1 = sym('x1','real');
x2 = sym('x2','real');

%% boundary function
% G = exp(sin(x1*x2/10)) + cos(x1)*sin(x2);
% G = sin(log(1+x1*x2));
% G = cos(x1)*sin(x2)*exp(sin(x1*x2));
G = sin(2*pi*x1)*cos(6*pi*x2);
g = matlabFunction(G);

%% forcing function
F = -diff(G,x1,2)-diff(G,x2,2);
f = matlabFunction(F);
% f = @(x1,x2) (40*pi^2) * sin(2.*pi.*x1).*cos(6.*pi.*x2);

%% mesh the rectange [0,1] x [2,3] with 20 / 60 subintervals in 
% x1-/x2-direction, respectively
msh = meshRectangle([0,1,2,3], [20,60]);

%% get discretized Poisson-Dirichlet problem
[A,b] = discretisePoisson(f,g,msh);

%% Preallocate u and set the boundary values
u = zeros(msh.N(2)+1,msh.N(1)+1);
u(1,:)         = g(msh.X1(1,:),         msh.X2(1,:));
u(end,:)       = g(msh.X1(end,:),       msh.X2(end,:));
u(2:end-1,1)   = g(msh.X1(2:end-1,1),   msh.X2(2:end-1,1));
u(2:end-1,end) = g(msh.X1(2:end-1,end), msh.X2(2:end-1,end));

%% solve problem for the interior points
u(2:end-1,2:end-1) = vec2msh(A\b, msh); % interior points

%% relative error
gh = g(msh.X1,msh.X2);
fprintf('max. error = %.6e\n', max(abs(vec(u-gh))));
fprintf('mean error = %.6e\n', mean(abs(vec(u-gh))));
fprintf('norm error = %.6e\n', norm(vec(u-gh)));
fprintf('rel. error = %.6e\n', norm(vec(u-gh))/norm(vec(gh)));

%% evaluate u on msh and draw a surface plot
figure, grid on
surf(msh.X1, msh.X2, u);

%% plot expected solution
figure, grid on
surf(msh.X1, msh.X2, gh);

%% axis labels
% textargs = {'fontsize',24,'interpreter','latex'};
% xlabel('$x_1$', textargs{:});
% ylabel('$x_2$', textargs{:});
% zlabel('$\sin(2\pi x_1)\cos(6\pi x_2)$', textargs{:});
% 
% % set background to white
% set(gcf,'color','w');
% 
% % save figure to pdf
% export_fig hw1_figure -transparent -pdf