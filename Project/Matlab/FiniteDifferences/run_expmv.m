function [ ] = run_expmv(rootpath)
%RUN_SPLIT

if nargin < 1; rootpath = '/home/jon/Documents/MATH521/Project/Matlab/FiniteDifferences'; end

T = 40e-3;
type = 'gre';
theta_deg = 90.0;

N_list = [50, 100, 150, 200, 250, 300];
prec_list = {'double','single','half'};
% vesselrad_list = [250.0, 125.0];
% vesselrad_list = [125.0];
vesselrad_list = [250.0];

for N = N_list
    for vesselrad = vesselrad_list
        [Geom, ~, ~, A] = bt_setup( N, vesselrad, theta_deg );
        run_split_iters(Geom, A, type, T, prec_list, rootpath);
    end
end

end

function run_split_iters(Geom, A, type, T, prec_list, rootpath)

x0 = 1i*ones(prod(Geom.GridSize(:)),1);
SignalScale = prod(Geom.VoxelSize)/prod(Geom.GridSize);

blank_rho = struct('order',[],'nrep',[],'h',[],'relerr',[],'ratio',[],'rmaxerr',[],'rsumerr',[],'isumerr',[],'time',[]);
blank_sigdata = struct('N', [], 'R', [], 'dt', [], 't', [], 'Sx', [], 'Sy', [], 'S', [], 'S_cplx_exact', NaN, 'walltime', []);
rho = [];
sigdata_all = [];

for ii = 1:numel(prec_list)
    dt = T;
    prec = prec_list{ii};
    
    t_step = tic;
    
    if strcmpi(type, 'gre')
        xh = expmv(T,A,x0,[],prec);
    else
        xh = expmv(T/2,A,x0,[],prec);
        xh = conj(xh);
        xh = expmv(T/2,A,x0,[],prec);
    end
    
    xh = reshape(xh, Geom.GridSize);
    Sh = SignalScale * sum(sum(sum(xh,1),2),3); % get signal
    
    t_step = toc(t_step);
    
    sigdata_all = cat(1,sigdata_all,blank_sigdata);
    sigdata_all(end).N = Geom.GridSize(1);
    sigdata_all(end).R = Geom.Rmajor;
    sigdata_all(end).dt = dt;
    sigdata_all(end).t = T;
    sigdata_all(end).Sx = real(Sh);
    sigdata_all(end).Sy = imag(Sh);
    sigdata_all(end).S = abs(Sh);
    sigdata_all(end).walltime = t_step;
    
    rho = cat(1,rho,blank_rho);
    rho(end).order = NaN;
    rho(end).nrep = NaN;
    rho(end).h = dt;
    rho(end).time = t_step;
    
    str = sprintf('N = %d, prec = %s, h = %1.3e, S = %1.8e', sigdata_all(end).N, prec, rho(end).h, sigdata_all(end).S);
    display_toc_time(t_step,str);
    
    savepath_base = [rootpath, '/expmv_results'];
    filepath_root = [savepath_base, '/N', num2str(Geom.GridSize(1)), '_r', num2str(round(Geom.Rmajor))];
    filepath = [filepath_root, '/', prec];
    filename = [filepath, '/sigdata'];
    
    if ~exist(filepath,'dir')
        mkdir(filepath);
    end
    sigdata = sigdata_all(end);
    save(filename,'sigdata');
    
end

end

