function [tr_data, be_data] = CompareBTSignals_FEM(rootpath)
%COMPAREBTSIGNALS_FEM

if nargin < 1; rootpath = '/home/jon/Documents/MATH521/Project'; end

[tr_data, be_data, ad_tr_data, ad_be_data, exact_data] = loadAllSignals(rootpath);

vec = @(x) x(:);
lastEntry = @(d,f) vec(cellfun(@(x)x(end),{d.(f)}));

s_tr = lastEntry(tr_data,'S');
s_be = lastEntry(be_data,'S');
s_ad_tr = lastEntry(ad_tr_data,'S');
s_ad_be = lastEntry(ad_be_data,'S');
t_tr = lastEntry(tr_data,'walltime');
t_be = lastEntry(be_data,'walltime');
t_ad_tr = lastEntry(ad_tr_data,'walltime');
t_ad_be = lastEntry(ad_be_data,'walltime');

exact_signal = abs(exact_data.S(end));

% ---- process (adaptive) euler/trbdf2 ---- %
err_be = abs(s_be - exact_signal)/exact_signal;
err_tr = abs(s_tr - exact_signal)/exact_signal;
err_ad_tr = abs(s_ad_tr - exact_signal)/exact_signal;
err_ad_be = abs(s_ad_be - exact_signal)/exact_signal;

% ---- plot figures ---- %
close all force; figure
plotargs = {'o--', 'markersize', 15, 'MarkerFaceColor', 'g', 'linewidth', 6};
h1 = loglog(t_ad_tr, err_ad_tr, plotargs{:}); hold on
h2 = loglog(t_ad_be, err_ad_be, plotargs{:});
h3 = loglog(t_tr, err_tr, plotargs{:});
h4 = loglog(t_be, err_be, plotargs{:});
leg = legend('Adaptive TRBDF-2','Adaptive Backward Euler','TRBDF-2','Backward Euler');

fontprops = {'fontsize', 30};
set(leg, fontprops{:});
xlabel('Simulation Time [s]', fontprops{:});
ylabel('Relative Error (Total Signal)', fontprops{:});

end

function [S_tr, S_be, S_ad_tr, S_ad_be, S_exact] = loadAllSignals(rootpath)

% ---- TRBDF2 data ---- %
tr_dt = {8e-3, 4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4};
dir_fun = @(dt) ['/Fenics/bt/results/union/cylinder_N200_ns64_r250/trbdf2/dt_', ...
    strrep(num2str(dt),'.','p'), '/signal.csv'];
tr_files = cellfun(dir_fun,tr_dt,'uniformoutput',false);
tr_files = strcat(rootpath, tr_files);
S_tr = loadSignals(tr_files, tr_dt);

% ---- TRBDF2 adaptive data ---- %
tr_iters = {0,1,2,3,4};
dir_fun = @(iter) ['/Fenics/bt/adapt/results/union/trbdf2/cylinder_N10_ns64_r250/Forw/dt_0p001/iter', ...
    num2str(iter), '/signal.csv'];
tr_files = cellfun(dir_fun,tr_iters,'uniformoutput',false);
tr_files = strcat(rootpath, tr_files);
S_ad_tr = loadSignals(tr_files, {0.001,0.001,0.001,0.001,0.001});

% ---- Backward Euler data ---- %
be_dt = {8e-3, 4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4};%, 6.25e-5, 3.125e-5};
dir_fun = @(dt) ['/Fenics/bt/results/union/cylinder_N200_ns64_r250/be/dt_', ...
    strrep(num2str(dt),'.','p'), '/signal.csv'];
be_files = cellfun(dir_fun,be_dt,'uniformoutput',false);
be_files = strcat(rootpath, be_files);
S_be = loadSignals(be_files, be_dt);

% ---- Backward Euler data ---- %
be_iters = {0,1,2,3,4};
dir_fun = @(iter) ['/Fenics/bt/adapt/results/union/be/cylinder_N10_ns64_r250/Forw/dt_0p001/iter', ...
    num2str(iter), '/signal.csv'];
be_files = cellfun(dir_fun,be_iters,'uniformoutput',false);
be_files = strcat(rootpath, be_files);
S_ad_be = loadSignals(be_files, {0.001,0.001,0.001,0.001,0.001});

% ---- Exact solution data ---- %
% exact_path = [rootpath, '/Fenics/bt/adapt/results/union/trbdf2/cylinder_N10_ns64_r250/Forw/dt_0p00025/iter4/signal.csv'];
% exact_path = [rootpath, '/Fenics/bt/adapt/results/union/be/cylinder_N10_ns64_r250/Forw/dt_0p00025/iter4/signal.csv'];
% exact_path = [rootpath, '/Fenics/bt/adapt/results/union/trbdf2/cylinder_N10_ns64_r250/Forw/dt_0p00025/exact/cylinder_N200_ns64_r250/signal.csv'];
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
