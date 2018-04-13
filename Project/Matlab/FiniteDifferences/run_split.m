function [ ] = run_split(rootpath)
%RUN_SPLIT 

if nargin < 1; rootpath = '/home/jon/Documents/MATH521/Project/Matlab/FiniteDifferences'; end

T = 40e-3;
type = 'gre';
theta_deg = 90.0;

N_list = [50, 100, 150, 200, 250, 300, 500];
dt_list = [40e-3, 20e-3, 8e-3, 4e-3, 2e-3, 1e-3];
vesselrad_list = [250.0, 125.0];

for N = N_list
    for vesselrad = vesselrad_list
        [Geom, Gamma, Dcoeff] = bt_setup( N, vesselrad, theta_deg );
        run_split_iters(Geom, Gamma, Dcoeff, type, T, dt_list, rootpath);
    end
end


end

function run_split_iters(Geom, Gamma, Dcoeff, type, T, dt_list, rootpath)

nrep_list = round(T./dt_list);

dGamma = {};
% xnorm = norm(vec(x));
% xmax = maxabs(x);
x0 = 1i*ones(Geom.GridSize);
SignalScale = prod(Geom.VoxelSize)/prod(Geom.GridSize);

blank_rho = struct('order',[],'nrep',[],'h',[],'relerr',[],'ratio',[],'rmaxerr',[],'rsumerr',[],'isumerr',[],'time',[]);
blank_sigdata = struct('N', [], 'R', [], 'dt', [], 't', [], 'Sx', [], 'Sy', [], 'S', [], 'S_cplx_exact', NaN, 'walltime', []);
rho = [];
sigdata = [];

for steporder = [2,4]
%     err_last = 0.0;
    for ii = 1:numel(nrep_list)
        nrep = nrep_list(ii);
        dt = dt_list(ii);
        
        t_step = tic;
        
        if strcmpi(type, 'gre')
            Vsub = SplittingMethods.BTSplitStepper( dt, Dcoeff, Gamma, dGamma, Geom.GridSize, Geom.VoxelSize, ...
                'Order', steporder, 'Nreps', nrep ); % for GRE
            xh = step(Vsub,x0);
        else
            Vsub = SplittingMethods.BTSplitStepper( dt, Dcoeff, Gamma, dGamma, Geom.GridSize, Geom.VoxelSize, ...
                'Order', steporder, 'Nreps', nrep/2 ); % for SE (Run twice)
            xh = step(Vsub,x0);
            xh = conj(xh);
            xh = step(Vsub,xh);
        end
        
        Sh = SignalScale * sum(sum(sum(xh,1),2),3); % get signal
        
        t_step = toc(t_step);
        
        sigdata = cat(1,sigdata,blank_sigdata);
        sigdata(end).N = Geom.GridSize(1);
        sigdata(end).R = Geom.Rmajor;
        sigdata(end).dt = dt;
        sigdata(end).t = T;
        sigdata(end).Sx = real(Sh);
        sigdata(end).Sy = imag(Sh);
        sigdata(end).S = abs(Sh);
        sigdata(end).walltime = t_step;
        
%         relerr  = norm(vec(x-xh))/xnorm;
%         ratio   = err_last/relerr;
%         rmaxerr = maxabs(x-xh)/xmax;
%         rsumerr = abs(real(S-Sh))/abs(real(S));
%         isumerr = abs(imag(S-Sh))/abs(imag(S));
%         
        rho = cat(1,rho,blank_rho);
        rho(end).order = steporder;
        rho(end).nrep = nrep;
        rho(end).h = dt;
%         rho(end).relerr = relerr;
%         rho(end).ratio = ratio;
%         rho(end).rmaxerr = rmaxerr;
%         rho(end).rsumerr = rsumerr;
%         rho(end).isumerr = isumerr;
        rho(end).time = t_step;
%         
%         str = sprintf('order = %d, nrep = %2d, h = %1.3e, relerr = %1.3e, ratio = %1.3e, relmax = %1.3e, rsumerr = %1.3e, isumerr = %1.3e', ...
%             rho(end).order, rho(end).nrep, rho(end).h, rho(end).relerr, rho(end).ratio, rho(end).rmaxerr, rho(end).rsumerr, rho(end).isumerr);
%         err_last = relerr;

        str = sprintf('N = %d, order = %d, nrep = %2d, h = %1.3e, S = %1.8e', sigdata(end).N, rho(end).order, rho(end).nrep, rho(end).h, sigdata(end).S);
        display_toc_time(t_step,str);
        
        savepath_base = [rootpath, '/split_results'];
        filepath_root = [savepath_base, '/N', num2str(Geom.GridSize(1)), '_r', num2str(round(Geom.Rmajor)), '_Order', num2str(steporder)];
        filepath = [filepath_root, '/dt_', strrep(num2str(dt),'.','p')];
        filename = [filepath, '/sigdata'];
        
        if ~exist(filepath,'dir')
            mkdir(filepath);
        end
        save(filename,'sigdata');
        
    end
    fprintf('\n');
end

end
