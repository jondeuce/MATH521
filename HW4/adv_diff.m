function [u_CD,u_UD,u_ex,ubar_ex,xgrid] = adv_diff(a,D,N)
%ADVECTION_DIFFUSION computes and plots the solution of the advection-
%diffusion equation
%       a u' - D u" = 0 on ]0,1[
%       u(0) = 0
%       u(1) = 1
%a is a real number, D a positive number and N is the (integer) number of
%subintervals for the mesh.
%
%Example of use:
%advection_diffusion(10,1,25)

h = 1/N;
ex = ones(N-1,1);
Dbar = a*h/2;

[f_CD,f_UD] = deal(zeros(N-1,1));
f_CD(end) = D/h^2 - a/(2*h);
f_UD(end) = D/h^2;

Dh = (D/h^2)*spdiags([-ex,2*ex,-ex],-1:1,N-1,N-1);
Ah_CD = (a/(2*h))*spdiags([-ex,ex],[-1,1],N-1,N-1);
Ah_UD = (a/h)*spdiags([-ex,ex],[-1,0],N-1,N-1);

xgrid = linspace(0,1,N+1)';
u_ex = (exp(a/D*xgrid) - 1)/(exp(a/D) - 1);
ubar_ex = (exp(a/(D+Dbar)*xgrid) - 1)/(exp(a/(D+Dbar)) - 1);

figure;
u_UD = [0;(Ah_UD+Dh)\f_UD;1];
u_CD = [0;(Ah_CD+Dh)\f_CD;1];
plot(xgrid,u_UD,'o-',xgrid,u_CD,'s-',xgrid,u_ex);
% plot(xgrid,log10(abs(u_UD)),'o-',xgrid,log10(abs(u_CD)),'s-',xgrid,log10(u_ex));
xlabel('{x}')
ylabel('{u}')
legend('upwind differencing','central differencing','analytical solution','location','northwest');
drawnow;
