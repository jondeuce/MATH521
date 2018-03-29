function energyplots
% Creates graphs with curves of the total energy E = T + V, the kinetic
% energy T, and the potential energy V as functions of time for the
% solution of the wave equation using three time stepping schemes, namely
% the implicit midpoint rule, the backward Euler method, and the forward
% Euler method.
% 
% The energy, potential energy, and kinetic energy are defined as follows:
%    E = T + V
%    T = 1/2 * ||v||^2
%    V = c^2/2 ||grad(u)||^2

% load data
cn = loadenergies('wave/cn/energy.csv'); save('cn','cn');
be = loadenergies('wave/be/energy.csv'); save('be','be');
fe = loadenergies('wave/fe/energy.csv'); save('fe','fe');

% create energy plots
plotenergies(cn, 'Implicit Midpoint Rule', 'cn');
plotenergies(be, 'Backward Euler Method', 'be');
plotenergies(fe, 'Forward Euler Method', 'fe');

end

function energydata = loadenergies(csvfilename)

energies = csvread(csvfilename);
energydata = struct(...
    't', energies(:,1), ...
    'E', energies(:,2), ...
    'T', energies(:,3), ...
    'V', energies(:,4) );

end

function plotenergies(data, method, figname)

% utility functions
vec = @(x) x(:);

% plot data
figure('name', figname), grid on
ydata = [data.E, data.T, data.V];
h = plot(data.t, ydata);

% plot prettifying
ylim([0, 1.01*max(vec(ydata))]);
title([method, ': Energy Functionals vs. Time']);
leg = legend(h, 'Total Energy', 'Kinetic Energy', 'Potential Energy');
set(leg, 'location', 'best');

% Save figures. For the export_fig package, see the MATLAB file exchange:
%    https://www.mathworks.com/matlabcentral/fileexchange/23629-export-fig
export_fig(figname, '-pdf', '-transparent');

end