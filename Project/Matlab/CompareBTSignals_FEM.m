function [tr_data, be_data] = CompareBTSignals_FEM(rootpath)
%COMPAREBTSIGNALS_FEM

if nargin < 1; rootpath = '/home/jon/Documents/MATH521/Project'; end

[tr_data, be_data, sp_data, expmv_data, exact_data] = loadAllSignals(rootpath);

vec = @(x) x(:);
lastEntry = @(d,f) vec(cellfun(@(x)x(end),{d.(f)}));

s_tr = lastEntry(tr_data,'S');
s_be = lastEntry(be_data,'S');
s_sp = lastEntry(sp_data,'S');
s_mv = lastEntry(expmv_data,'S');
t_tr = lastEntry(tr_data,'walltime');
t_be = lastEntry(be_data,'walltime');
t_sp = lastEntry(sp_data,'walltime');
t_mv = lastEntry(expmv_data,'walltime');

exact_signal = abs(exact_data.S(end));

% ---- process euler/trbdf2/split/expmv ---- %
err_be = abs(s_be - exact_signal)/exact_signal;
err_tr = abs(s_tr - exact_signal)/exact_signal;
err_sp = abs(s_sp - exact_signal)/exact_signal;
err_mv = abs(s_mv - exact_signal)/exact_signal;

% ---- plot figures ---- %
close all force; figure
plotargs = {'o--', 'markersize', 15, 'MarkerFaceColor', 'g', 'linewidth', 6};
h1 = loglog(t_sp, err_sp, plotargs{:}); hold on
h2 = loglog(t_mv, err_mv, plotargs{:});
h3 = loglog(t_tr, err_tr, plotargs{:});
h4 = loglog(t_be, err_be, plotargs{:});
leg = legend('Splitting Method','expmv','TRBDF-2','Backward Euler');

fontprops = {'fontsize', 30};
set(leg, fontprops{:});
xlabel('Simulation Time [s]', fontprops{:});
ylabel('Relative Error (Total Signal)', fontprops{:});

end

function [S_tr, S_be, S_sp, S_expmv, S_exact] = loadAllSignals(rootpath)

% ---- TRBDF2 data ---- %
tr_dt = {8e-3, 4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4};
dir_fun = @(dt) ['/Fenics/bt/results/union/cylinder_N200_ns64_r250/trbdf2/dt_', ...
    strrep(num2str(dt),'.','p'), '/signal.csv'];
tr_files = cellfun(dir_fun,tr_dt,'uniformoutput',false);
tr_files = strcat(rootpath, tr_files);
S_tr = loadSignals(tr_files, tr_dt);

% ---- Backward Euler data ---- %
be_dt = {8e-3, 4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4};%, 6.25e-5, 3.125e-5};
dir_fun = @(dt) ['/Fenics/bt/results/union/cylinder_N200_ns64_r250/be/dt_', ...
    strrep(num2str(dt),'.','p'), '/signal.csv'];
be_files = cellfun(dir_fun,be_dt,'uniformoutput',false);
be_files = strcat(rootpath, be_files);
S_be = loadSignals(be_files, be_dt);

% ---- Splitting data ---- %
sp_dt = {40e-3, 20e-3, 8e-3, 4e-3, 2e-3, 1e-3};
dir_fun = @(dt) ['/Matlab/FiniteDifferences/split_results/N200_r250_Order2/dt_', strrep(num2str(dt),'.','p'), '/sigdata.mat'];
sp_files = cellfun(dir_fun,sp_dt,'uniformoutput',false);
sp_files = strcat(rootpath, sp_files);

% S_sp = loadSignals(sp_files, sp_dt)
S_sp = [];
for f = sp_files
    s_sp = load(f{1});
    S_sp = cat(2, S_sp, s_sp.sigdata(end));
end

% ---- Expmv data ---- %
S_expmv = [];
for N = 50:50:300
    sp_path = [rootpath, '/Matlab/FiniteDifferences/expmv_results/N', num2str(N), '_r250/double'];
    s_sp = load([sp_path, '/sigdata.mat']);
    S_expmv = cat(2, S_expmv, s_sp.sigdata);
end

% ---- Exact solution data ---- %
% exact_path = [rootpath, '/Fenics/bt/adapt/results/union/trbdf2/cylinder_N10_ns64_r250/Forw/dt_0p00025/iter4/signal.csv'];
% exact_path = [rootpath, '/Fenics/bt/adapt/results/union/be/cylinder_N10_ns64_r250/Forw/dt_0p00025/iter4/signal.csv'];
% exact_path = [rootpath, '/Fenics/bt/adapt/results/union/trbdf2/cylinder_N10_ns64_r250/Forw/dt_0p00025/exact/cylinder_N200_ns64_r250/signal.csv'];
% exact_path = [rootpath, '/Fenics/bt/adapt/results/union/be/cylinder_N10_ns64_r250/Forw/dt_0p001/exact/cylinder_N250_ns64_r250/signal.csv'];
exact_path = [rootpath, '/Fenics/bt/adapt/results/union/trbdf2/cylinder_N10_ns64_r250/Forw/dt_0p001/exact/cylinder_N250_ns64_r250/signal.csv'];
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
