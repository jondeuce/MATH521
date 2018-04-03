%% SplittingMethodsTest()

%% Test parameters
% Glen = 256; % Grid side length [voxels]
% Vlen = 3000; % Voxel side length [um]
% Scale = Vlen/Glen;
% TypicalScale = 3000/512; % Typical scale [um/voxel]
% Gsize = Glen * [1,1,1];
% Vsize = Vlen * [1,1,1];

ScaleGsize = 2;
Gsize = [350,350,350] / ScaleGsize;
Vsize = [1750,1750,1750];
% Gsize = [350,350,800] / ScaleGsize;
% Vsize = [1750,1750,4000];
TypicalScale = 4000/800;
Scale = Vsize(3)/Gsize(3);

t = 40e-3;
type = 'gre';
% t = 60e-3;
% type = 'se';
Dcoeff = 3037 * (Scale/TypicalScale)^2; % Scale diffusion to mimic [3000 um]^3 512^3 grid

Rminor_mu  = 25;
Rminor_sig = 0;
% Rminor_mu  = 13.7;
% Rminor_sig = 2.1;
% Rminor_mu  = 7;
% Rminor_sig = 0;

iBVF = 1.1803/100;
aBVF = 1.3425/100;
Nmajor = 4; % Number of major vessels (optimal number is from SE perf. orientation. sim)
MajorAngle = 0.0; % Angle of major vessels
NumMajorArteries = 1; % Number of major arteries
MinorArterialFrac = 1/3; % Fraction of minor vessels which are arteries
rng('default'); seed = rng;

%% Calculate Geometry
GammaSettings = Geometry.ComplexDecaySettings('Angle_Deg', 90, 'B0', -3);
Geom = Geometry.CylindricalVesselFilledVoxel( ...
    'iBVF', iBVF, 'aBVF', aBVF, ...
    'VoxelSize', Vsize, 'GridSize', Gsize, 'VoxelCenter', [0,0,0], ...
    'Nmajor', Nmajor, 'MajorAngle', MajorAngle, ...
    'NumMajorArteries', NumMajorArteries, 'MinorArterialFrac', MinorArterialFrac, ...
    'Rminor_mu', Rminor_mu, 'Rminor_sig', Rminor_sig, ...
    'AllowMinorSelfIntersect', true, 'AllowMinorMajorIntersect', true, ...
    'PopulateIdx', true, 'seed', seed );

%% Calculate ComplexDecay
Gamma = CalculateComplexDecay(GammaSettings, Geom);
dGamma = {};

%% Initial data
% x0 = randnc(Gsize);
x0 = 1i*ones(Gsize);
S0 = 1i*numel(x0);

%% Calculate exact solution via expmv
t_expmv = tic;

A = BlochTorreyOp(Gamma, Dcoeff, Gsize, Vsize);
% x = bt_expmv( t, A, x0, 'prnt', false, 'type', type );

A = sparse(A);
x = expmv(t,A,x0(:),[],'double');
x = reshape(x,size(x0));

S = sum(sum(sum(x,1),2),3); %more accurate than sum(x(:))

t_expmv = toc(t_expmv);
display_toc_time(t_expmv,'expmv');

%% BTStepper Test Loop
xnorm = norm(vec(x));
xmax = maxabs(x);
blank_rho = struct('order',[],'nrep',[],'h',[],'relerr',[],'ratio',[],'rmaxerr',[],'rsumerr',[],'isumerr',[],'time',[]);
blank_sigdata = struct('dt', [], 't', [], 'Sx', [], 'Sy', [], 'S', [], 'S_cplx_exact', S, 'walltime', []);

rho = [];
sigdata = [];
for steporder = 2%[2,4]
    if strcmpi(type, 'gre')
        if steporder == 2; nreps = 2.^(0:6); end % for GRE
        if steporder == 4; nreps = 2.^(0:4); end % for GRE
    else
        if steporder == 2; nreps = 2.^(1:6); end % for SE
        if steporder == 4; nreps = 2.^(1:4); end % for SE
    end
    err_last = 0.0;
    for ii = 1:numel(nreps)
        nrep = nreps(ii);
        t_step = tic;
        
        dt = t/nrep;
        if strcmpi(type, 'gre')
            Vsub = SplittingMethods.BTSplitStepper( dt, Dcoeff, Gamma, dGamma, Gsize, Vsize, ...
                'Order', steporder, 'Nreps', nrep ); % for GRE
            xh = step(Vsub,x0);
        else
            Vsub = SplittingMethods.BTSplitStepper( dt, Dcoeff, Gamma, dGamma, Gsize, Vsize, ...
                'Order', steporder, 'Nreps', nrep/2 ); % for SE
            xh = step(Vsub,x0);
            xh = conj(xh);
            xh = step(Vsub,xh);
        end
        
        Sh = sum(sum(sum(xh,1),2),3); % get signal
        
        t_step = toc(t_step);
        
        sigdata = cat(1,sigdata,blank_sigdata);
        sigdata(end).dt = dt;
        sigdata(end).t = t;
        sigdata(end).Sx = real(Sh);
        sigdata(end).Sy = imag(Sh);
        sigdata(end).S = abs(Sh);
        sigdata(end).walltime = t_step;
        
        relerr  = norm(vec(x-xh))/xnorm;
        ratio   = err_last/relerr;
        rmaxerr = maxabs(x-xh)/xmax;
        rsumerr = abs(real(S-Sh))/abs(real(S));
        isumerr = abs(imag(S-Sh))/abs(imag(S));
        
        rho = cat(1,rho,blank_rho);
        rho(end).order = steporder;
        rho(end).nrep = nrep;
        rho(end).h = dt;
        rho(end).relerr = relerr;
        rho(end).ratio = ratio;
        rho(end).rmaxerr = rmaxerr;
        rho(end).rsumerr = rsumerr;
        rho(end).isumerr = isumerr;
        rho(end).time = t_step;
        
        str = sprintf('order = %d, nrep = %2d, h = %1.3e, relerr = %1.3e, ratio = %1.3e, relmax = %1.3e, rsumerr = %1.3e, isumerr = %1.3e', ...
            rho(end).order, rho(end).nrep, rho(end).h, rho(end).relerr, rho(end).ratio, rho(end).rmaxerr, rho(end).rsumerr, rho(end).isumerr);
        display_toc_time(t_step,str);
        err_last = relerr;
    end
    fprintf('\n');
end

%% expmv Test Loop
xnorm = norm(vec(x));
xmax = maxabs(x);
blank_res = struct('prec',[],'nrep',[],'h',[],'relerr',[],'ratio',[],'rmaxerr',[],'rsumerr',[],'isumerr',[],'time',[]);
res = [];
for prec = {'half','single'} %,'double'}
    err_last = 0.0;
    if strcmpi(type,'gre')
        nreps = 2.^(0:3); % for gre
    else
        nreps = 2.^(1:3); % for SE
    end
    for ii = 1:numel(nreps)
        nrep = nreps(ii);
        t_step = tic;
        
        dt = t/nrep;
        xh = bt_expmv_nsteps( dt, A, x0, nrep, 'calcsignal', 'none', 'type', type, 'prec', prec{1}, 'prnt', false );
        Sh = sum(sum(sum(xh,1),2),3);
        
        t_step = toc(t_step);
        
        relerr  = norm(vec(x-xh))/xnorm;
        ratio   = err_last/relerr;
        rmaxerr = maxabs(x-xh)/xmax;
        rsumerr = abs(real(S-Sh))/abs(real(S));
        isumerr = abs(imag(S-Sh))/abs(imag(S));
        
        res = cat(1,res,blank_res);
        res(end).prec = prec{1};
        res(end).nrep = nrep;
        res(end).h = dt;
        res(end).relerr = relerr;
        res(end).ratio = ratio;
        res(end).rmaxerr = rmaxerr;
        res(end).rsumerr = rsumerr;
        res(end).isumerr = isumerr;
        res(end).time = t_step;
        
        str = sprintf('prec = %s, nrep = %2d, h = %1.3e, relerr = %1.3e, ratio = %1.3e, relmax = %1.3e, rsumerr = %1.3e, isumerr = %1.3e', ...
            res(end).prec, res(end).nrep, res(end).h, res(end).relerr, res(end).ratio, res(end).rmaxerr, res(end).rsumerr, res(end).isumerr);
        display_toc_time(t_step,str);
        err_last = relerr;
    end
    fprintf('\n');
end
