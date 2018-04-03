function [tr_data, be_data] = CompareBTSignals

[tr_data, be_data, sp_data] = loadAllSignals;

vec = @(x) x(:);
lastEntry = @(d,f) vec(cellfun(@(x)x(end),{d.(f)}));

% ---- process euler/trbdf2 ---- %
S_tr = lastEntry(tr_data,'S');
S_be = lastEntry(be_data,'S');
T_tr = lastEntry(tr_data,'walltime');
T_be = lastEntry(be_data,'walltime');

exact_signal = S_tr(end);
s_tr = S_tr(1:end-1);
t_tr = T_tr(1:length(s_tr));
s_be = S_be(1:length(s_tr));
t_be = T_be(1:length(s_tr));

err_tr = abs(s_tr - exact_signal)/exact_signal;
err_be = abs(s_be - exact_signal)/exact_signal;

% ---- process splitting method ---- %
exact_cplx_signal = sp_data(1).S_cplx_exact;
exact_signal = abs(exact_cplx_signal);
s_sp = lastEntry(sp_data,'S');
t_sp = lastEntry(sp_data,'walltime');

err_sp = abs(s_sp - exact_signal)/exact_signal;

close all force; figure
plotargs = {'o--', 'markersize', 15, 'MarkerFaceColor', 'g', 'linewidth', 6};
h1 = loglog(t_sp, err_sp, plotargs{:}); hold on
h2 = loglog(t_tr, err_tr, plotargs{:});
h3 = loglog(t_be, err_be, plotargs{:});
leg = legend('Splitting Method','TRBDF2','Backward Euler');

fontprops = {'fontsize', 30};
set(leg, fontprops{:});
xlabel('Simulation Time [s]', fontprops{:});
ylabel('Relative Error (Total Signal)', fontprops{:});

end

function [S_tr, S_be, S_sp] = loadAllSignals

% ---- TRBDF2 data ---- %
tr_files = {'Fenics/trbdf2/tmp/dt_8e-3/signal.csv',...
    'Fenics/trbdf2/tmp/dt_4e-3/signal.csv',...
    'Fenics/trbdf2/tmp/dt_2e-3/signal.csv',...
    'Fenics/trbdf2/tmp/dt_1e-3/signal.csv',...
    'Fenics/trbdf2/tmp/dt_5e-4/signal.csv',...
    'Fenics/trbdf2/tmp/dt_2p5e-4/signal.csv',...
    'Fenics/trbdf2/tmp/dt_1p25e-4/signal.csv'};

tr_dt = {8e-3, 4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4};

% ---- Backward Euler data ---- %
be_files = {'Fenics/be/tmp/dt_8e-3/signal.csv',...
    'Fenics/be/tmp/dt_4e-3/signal.csv',...
    'Fenics/be/tmp/dt_2e-3/signal.csv',...
    'Fenics/be/tmp/dt_1e-3/signal.csv',...
    'Fenics/be/tmp/dt_5e-4/signal.csv',...
    'Fenics/be/tmp/dt_2p5e-4/signal.csv',...
    'Fenics/be/tmp/dt_1p25e-4/signal.csv',...
    'Fenics/be/tmp/dt_6p25e-5/signal.csv',...
    'Fenics/be/tmp/dt_3p125e-5/signal.csv'};

be_dt = {8e-3, 4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5, 3.125e-5};

S_tr = loadSignals(tr_files, tr_dt);
S_be = loadSignals(be_files, be_dt);

% ---- Splitting Method data ---- %
S_sp = load('Matlab/SplittingMethods/Split_175iso.mat');
S_sp = S_sp.sigdata;

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
