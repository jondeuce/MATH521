% Investigating the effect of the step size h on the solution of the
% advection-diffusion equation in 1D using upwind and centered difference
% schemes for the first derivative
close all force; clear all; clc

%% Declare test variables
Ncrit = 128; % Ncrit = 1/hcrit = a/2D
D = pi; % Choose D arbitrarily
a = 2*D*Ncrit; % choose 'a' such that hcrit = 2D/a = 1/Ncrit above

% Set the range of N values to be = [4, 8, 16, ..., Ncrit, ..., Ncrit^2]
% (Ncrit is assumed to have been chosen to be a power of 2)
pmin = 2;
pmax = 2*round(log2(Ncrit));
Nrange = 2.^(pmin:pmax);

%% Solve the system below and above the critical point hcrit = 2D/a
for N = Nrange
    advection_diffusion(a,D,N);
    drawnow; title(sprintf('N = %d',N));
end
