n = 100;
A = randSPDmat(n);
% M = eye(n);
M = randSPDmat(n);
u0 = randn(n,1);

nsteps = 320;
tspan(2) = 1/(max(abs(eig(M)))/min(abs(eig(A))));
dt = diff(tspan)/nsteps;

% n = 100;
% A = randSPDmat(n);
% M = randSPDmat(n);

tic; [t1,u1] = ode45(@(~,u)-(A*u),tspan,u0,odeset('mass',M,'abstol',1e-14,'reltol',1e-12)); toc;
tic; [t2,u2] = trbdf2({A,M,dt},tspan,u0); toc;

close all
% figure, plot(t1,u1)
% figure, plot(t2,u2)
figure, plot(u1(end,:))
figure, plot(u2(end,:))
disp(norm(u1(end,:) - u2(end,:)))
disp(maxabs(u1(end,:) - u2(end,:)))
