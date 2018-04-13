function [tr_u_data, be_u_data] = CompareBTSignals_UnionVsHollow(rootpath)
%COMPAREBTSIGNALS_UNIONVSHOLLOW

if nargin < 1; rootpath = '/home/jon/Documents/MATH521/Project'; end

[tr_u_data, be_u_data, tr_h_data, be_h_data, exact_data] = loadAllSignals(rootpath);

vec = @(x) x(:);
lastEntry = @(d,f) vec(cellfun(@(x)x(end),{d.(f)}));

s_tr_u = lastEntry(tr_u_data,'S');
s_be_u = lastEntry(be_u_data,'S');
t_tr_u = lastEntry(tr_u_data,'walltime');
t_be_u = lastEntry(be_u_data,'walltime');
t_tr_h = lastEntry(tr_h_data,'walltime');
t_be_h = lastEntry(be_h_data,'walltime');

% since initial signal is constant (0,1), volume of interior is equal to difference in initial signals
InteriorVolume = be_u_data(1).Sy(1) - be_h_data(1).Sy(1);
g = 267515255; %gyro mag
B0 = -3; %mag field
theta_deg = 90.0; %field angle
chi = 3.87816e-07; %magnetic susceptibilitys
a2 = 250.0^2; %vessel radius squared
omega_blood = chi*g*B0/6*(3*cosd(theta_deg)^2-1);
r2decay_blood = 30.0125;
gamma_blood = complex(r2decay_blood, omega_blood);
T = 40e-3;

shift = exp(-T*gamma_blood) * InteriorVolume;

s_tr_h = abs(complex(lastEntry(tr_h_data,'Sx'),lastEntry(tr_h_data,'Sy')) + shift);
s_be_h = abs(complex(lastEntry(be_h_data,'Sx'),lastEntry(be_h_data,'Sy')) + shift);
exact_signal = abs(complex(exact_data.Sx(end),exact_data.Sy(end)) + shift);

% ---- process euler/trbdf2/split/expmv ---- %
err_be_u = abs(s_be_u - exact_signal)/exact_signal;
err_tr_u = abs(s_tr_u - exact_signal)/exact_signal;
err_tr_h = abs(s_tr_h - exact_signal)/exact_signal;
err_be_h = abs(s_be_h - exact_signal)/exact_signal;

% ---- plot figures ---- %
close all force; figure
plotargs = {'o--', 'markersize', 15, 'MarkerFaceColor', 'g', 'linewidth', 6};
h1 = loglog(t_tr_h, err_tr_h, plotargs{:}); hold on
h2 = loglog(t_be_h, err_be_h, plotargs{:});
h3 = loglog(t_tr_u, err_tr_u, plotargs{:});
h4 = loglog(t_be_u, err_be_u, plotargs{:});
leg = legend('Hollow: TRBDF-2','Hollow: Backward Euler','Union: TRBDF-2','Union: Backward Euler');

fontprops = {'fontsize', 30};
set(leg, fontprops{:});
xlabel('Simulation Time [s]', fontprops{:});
ylabel('Relative Error (Total Signal)', fontprops{:});

end

function [S_tr_u, S_be_u, S_tr_h, S_be_h, S_exact] = loadAllSignals(rootpath)

% ---- TRBDF2 data (union) ---- %
tr_u_dt = {8e-3, 4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4};
dir_fun = @(dt) ['/Fenics/bt/results/union/cylinder_N200_ns64_r250/trbdf2/dt_', ...
    strrep(num2str(dt),'.','p'), '/signal.csv'];
tr_u_files = cellfun(dir_fun,tr_u_dt,'uniformoutput',false);
tr_u_files = strcat(rootpath, tr_u_files);
S_tr_u = loadSignals(tr_u_files, tr_u_dt);

% ---- Backward Euler data (union) ---- %
be_u_dt = {8e-3, 4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4};%, 6.25e-5, 3.125e-5};
dir_fun = @(dt) ['/Fenics/bt/results/union/cylinder_N200_ns64_r250/be/dt_', ...
    strrep(num2str(dt),'.','p'), '/signal.csv'];
be_u_files = cellfun(dir_fun,be_u_dt,'uniformoutput',false);
be_u_files = strcat(rootpath, be_u_files);
S_be_u = loadSignals(be_u_files, be_u_dt);

% ---- TRBDF2 data (hollow) ---- %
tr_h_dt = {8e-3, 4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4};
dir_fun = @(dt) ['/Fenics/bt/results/hollow/cylinder_N200_ns64_r250/trbdf2/dt_', ...
    strrep(num2str(dt),'.','p'), '/signal.csv'];
tr_h_files = cellfun(dir_fun,tr_h_dt,'uniformoutput',false);
tr_h_files = strcat(rootpath, tr_h_files);
S_tr_h = loadSignals(tr_h_files, tr_h_dt);

% ---- Backward Euler data (hollow) ---- %
be_h_dt = {8e-3, 4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4};%, 6.25e-5, 3.125e-5};
dir_fun = @(dt) ['/Fenics/bt/results/hollow/cylinder_N200_ns64_r250/be/dt_', ...
    strrep(num2str(dt),'.','p'), '/signal.csv'];
be_h_files = cellfun(dir_fun,be_h_dt,'uniformoutput',false);
be_h_files = strcat(rootpath, be_h_files);
S_be_h = loadSignals(be_h_files, be_h_dt);

% ---- Exact solution data ---- %
% exact_path = [rootpath, '/Fenics/bt/adapt/results/union/trbdf2/cylinder_N10_ns64_r250/Forw/dt_0p00025/iter4/signal.csv'];
% exact_path = [rootpath, '/Fenics/bt/adapt/results/union/be/cylinder_N10_ns64_r250/Forw/dt_0p00025/iter4/signal.csv'];
% exact_path = [rootpath, '/Fenics/bt/adapt/results/union/trbdf2/cylinder_N10_ns64_r250/Forw/dt_0p00025/exact/cylinder_N200_ns64_r250/signal.csv'];
% exact_path = [rootpath, '/Fenics/bt/adapt/results/union/be/cylinder_N10_ns64_r250/Forw/dt_0p001/exact/cylinder_N250_ns64_r250/signal.csv'];
% exact_path = [rootpath, '/Fenics/bt/adapt/results/union/trbdf2/cylinder_N10_ns64_r250/Forw/dt_0p001/exact/cylinder_N250_ns64_r250/signal.csv'];
exact_path = [rootpath, '/Fenics/bt/adapt/results/hollow/trbdf2/cylinder_N10_ns64_r250/Forw/dt_0p001/exact/cylinder_N250_ns64_r250/signal.csv'];
S_exact = loadSignals({exact_path}, {2.5e-4});

end


function S = loadSignals(fileList, dt)

S = struct('dt', [], 't', [], 'Sx', [], 'Sy', [], 'S', [], 'walltime', []);
for ii = 1:length(fileList)
    S(ii).dt = dt{ii};
    
    sig = load(fileList{ii});
    S(ii).t = sig(:,1);
    S(ii).Sx = sig(:,2);
    S(ii).Sy = sig(:,3);
    S(ii).S = sig(:,4);
    S(ii).walltime = sig(:,5);
end

end
