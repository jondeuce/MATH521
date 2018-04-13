function [sp100_data, sp200_data] = CompareBTSignals_Expmv(rootpath)
%COMPAREBTSIGNALS_EXPMV

if nargin < 1; rootpath = '/home/jon/Documents/MATH521/Project'; end

[sp100_data, sp200_data, sp300_data, mvHalf_data, mvSing_data, mvDoub_data, exact_data] = loadAllSignals(rootpath);
% S_sp_N100, S_sp_N200, S_sp_N300, S_expmv_half, S_expmv_single, S_expmv_double, S_exact

vec = @(x) x(:);
lastEntry = @(d,f) vec(cellfun(@(x)x(end),{d.(f)}));

s_sp100 = lastEntry(sp100_data,'S');
s_sp200 = lastEntry(sp200_data,'S');
s_sp300 = lastEntry(sp300_data,'S');
s_mvHalf = lastEntry(mvHalf_data,'S');
s_mvSing = lastEntry(mvSing_data,'S');
s_mvDoub = lastEntry(mvDoub_data,'S');

t_sp100 = lastEntry(sp100_data,'walltime');
t_sp200 = lastEntry(sp200_data,'walltime');
t_sp300 = lastEntry(sp300_data,'walltime');
t_mvHalf = lastEntry(mvHalf_data,'walltime');
t_mvSing = lastEntry(mvSing_data,'walltime');
t_mvDoub = lastEntry(mvDoub_data,'walltime');

exact_signal = abs(exact_data.S(end));

% ---- process euler/trbdf2/split/expmv ---- %
err_sp100 = abs(s_sp100 - exact_signal)/exact_signal;
err_sp200 = abs(s_sp200 - exact_signal)/exact_signal;
err_sp300 = abs(s_sp300 - exact_signal)/exact_signal;
err_mvHalf = abs(s_mvHalf - exact_signal)/exact_signal;
err_mvSing = abs(s_mvSing - exact_signal)/exact_signal;
err_mvDoub = abs(s_mvDoub - exact_signal)/exact_signal;

% ---- plot figures ---- %
close all force; figure
plotargs = {'o--', 'markersize', 15, 'MarkerFaceColor', 'g', 'linewidth', 6};
h1 = loglog(t_mvHalf, err_mvHalf, plotargs{:}); hold on
h2 = loglog(t_mvSing, err_mvSing, plotargs{:});
h3 = loglog(t_mvDoub, err_mvDoub, plotargs{:});
h4 = loglog(t_sp100, err_sp100, plotargs{:});
h5 = loglog(t_sp200, err_sp200, plotargs{:});
h6 = loglog(t_sp300, err_sp300, plotargs{:});
leg = legend('expmv: Half Precision','expmv: Single Precision','expmv: Double Precision',...
    'Splitting Method: N = 100', 'Splitting Method: N = 200', 'Splitting Method: N = 300');

fontprops = {'fontsize', 30};
set(leg, fontprops{:});
xlabel('Simulation Time [s]', fontprops{:});
ylabel('Relative Error (Total Signal)', fontprops{:});

end

function [S_sp_N100, S_sp_N200, S_sp_N300, S_expmv_half, S_expmv_single, S_expmv_double, S_exact] = loadAllSignals(rootpath)

% ---- Splitting data ---- %
sp_dt = {40e-3, 20e-3, 8e-3, 4e-3, 2e-3, 1e-3};
dir_fun = @(N,dt) ['/Matlab/FiniteDifferences/split_results/N', num2str(N), ...
    '_r250_Order2/dt_', strrep(num2str(dt),'.','p'), '/sigdata.mat'];
sp_files_fun = @(N) cellfun(@(dt)dir_fun(N,dt),sp_dt,'uniformoutput',false);

sp_files = strcat(rootpath, sp_files_fun(100));
S_sp_N100 = [];
for f = sp_files
    s_sp = load(f{1});
    S_sp_N100 = cat(2, S_sp_N100, s_sp.sigdata(end));
end
sp_files = strcat(rootpath, sp_files_fun(200));
S_sp_N200 = [];
for f = sp_files
    s_sp = load(f{1});
    S_sp_N200 = cat(2, S_sp_N200, s_sp.sigdata(end));
end
sp_files = strcat(rootpath, sp_files_fun(300));
S_sp_N300 = [];
for f = sp_files
    s_sp = load(f{1});
    S_sp_N300 = cat(2, S_sp_N300, s_sp.sigdata(end));
end

% ---- Expmv data ---- %
S_expmv_double = [];
for N = 50:50:250
    sp_path = [rootpath, '/Matlab/FiniteDifferences/expmv_results/N', num2str(N), '_r250/double'];
    s_sp = load([sp_path, '/sigdata.mat']);
    S_expmv_double = cat(2, S_expmv_double, s_sp.sigdata);
end
S_expmv_single = [];
for N = 50:50:300
    sp_path = [rootpath, '/Matlab/FiniteDifferences/expmv_results/N', num2str(N), '_r250/single'];
    s_sp = load([sp_path, '/sigdata.mat']);
    S_expmv_single = cat(2, S_expmv_single, s_sp.sigdata);
end
S_expmv_half = [];
for N = 50:50:300
    sp_path = [rootpath, '/Matlab/FiniteDifferences/expmv_results/N', num2str(N), '_r250/half'];
    s_sp = load([sp_path, '/sigdata.mat']);
    S_expmv_half = cat(2, S_expmv_half, s_sp.sigdata);
end

% ---- Exact solution data ---- %
sp_path = [rootpath, '/Matlab/FiniteDifferences/expmv_results/N300_r250/double'];
s_sp = load([sp_path, '/sigdata.mat']);
S_exact = s_sp.sigdata;

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
