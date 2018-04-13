function [ Geom, Gamma, Dcoeff, A ] = bt_setup( N, vesselrad, theta_deg )
%GET_GEOM Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3; theta_deg = 90.0; end
if nargin < 2; vesselrad = 250.0; end
if nargin < 1; N = 250; end

%% setup params
L = 3000;
Gsize = [N,N,N];
Vsize = [L,L,L];

t = 40e-3;
type = 'gre';
Dcoeff = 3037;
B0_Angle_Deg = theta_deg;
B0 = -3.0;

Rminor_mu  = 250;
Rminor_sig = 0;
% Rminor_mu  = 13.7;
% Rminor_sig = 2.1;
% Rminor_mu  = 7;
% Rminor_sig = 0;

iBVF = 10.0/100;
aBVF = 10.0/100;
Nmajor = 1; % Number of major vessels (optimal number is from SE perf. orientation. sim)
MajorAngle = 0.0; % Angle of major vessels
NumMajorArteries = 0; % Number of major arteries
MinorArterialFrac = 0.0; % Fraction of minor vessels which are arteries
rng('default'); seed = rng;

%% Calculate Geometry
Geom = Geometry.CylindricalVesselFilledVoxel( ...
    'iBVF', iBVF, 'aBVF', aBVF, ...
    'VoxelSize', Vsize, 'GridSize', Gsize, 'VoxelCenter', [0,0,0], ...
    'Nmajor', Nmajor, 'MajorAngle', MajorAngle, ...
    'NumMajorArteries', NumMajorArteries, 'MinorArterialFrac', MinorArterialFrac, ...
    'Rminor_mu', Rminor_mu, 'Rminor_sig', Rminor_sig, ...
    'AllowMinorSelfIntersect', true, 'AllowMinorMajorIntersect', true, ...
    'PopulateIdx', true, 'Verbose', false, 'seed', seed );
Geom = SetCylinders( Geom, [0;0;0], vesselrad, [0;0;1], [], [], [] );

%% Calculate Gamma
GammaSettings = Geometry.ComplexDecaySettings('Angle_Deg', B0_Angle_Deg, 'B0', B0);
% Gamma = CalculateComplexDecay(GammaSettings, Geom, 'Gamma');

r2decay = CalculateComplexDecay(GammaSettings, Geom, 'r2');

[X,Y] = meshgrid(linspacePeriodic(-L/2,L/2,N));
X2 = X.*X; Y2 = Y.*Y; R4 = (X2+Y2).^2;
chi = GammaSettings.dChi_Blood;
a2 = vesselrad^2;
g = GammaSettings.GyroMagRatio;
omega = Geom.VasculatureMap * (chi*g*B0/6*(3*cosd(theta_deg)^2-1)) + ... % interior
    bsxfun(@times, (1-Geom.VasculatureMap), ...
    (chi*g*B0/2*sind(theta_deg)^2)*(a2*(Y2-X2)./R4)); % exterior

Gamma = complex(r2decay, omega);

%% Get Bloch-Torrey operator A as a sparse matrix
if nargout > 3
    A = BlochTorreyOp(Gamma, Dcoeff, Gsize, Vsize);
    A = sparse(A);
end

end

