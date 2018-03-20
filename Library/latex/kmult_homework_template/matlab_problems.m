%% Initialization of Constants
% Unit conversions and physical constants -- SI

fm = 1E-15; % m to fm
MeVtoJ = 1.6021766E-13; % MeV to J
csq = 8.988E16; % c^2
hbarsq = 1.112122E-68; % reduced planck's constant squared

% Calculation of reduced mass and binding energy in SI
mu = 469.46 * MeVtoJ / csq; % kg
B = 2.22 * MeVtoJ; % J

% Square-Well Parameters
R = 2.0 * fm; % size of finite square-well potential
V0 = 37.0 * MeVtoJ; % height of potential well

% Calculating mode and decay constant of wave function
k = sqrt(2*mu/hbarsq * (V0-B) );
gamma = sqrt( 2*mu/hbarsq * B );

% Calculating normalization constants
Asq = 2*gamma/(1 + gamma*R);
Csq = Asq * sin(k*R)^2 * exp(2*gamma*R);
A = sqrt(Asq);
C = sqrt(Csq);

%% Plotting Deuteron Wave Function
r = linspace(0, 15, 1000) * fm; % sample points (in meters)

% defining the wave-function inside and outside of the well
u_in = sprintf('%f * sin(%f * r)', A, k); 
u_out = sprintf('%f * exp(-%f * r)', C, gamma); 

% total wave-function
u = @(r) piecewise_eval(r , R, {u_in,u_out});

plot(r/fm, u(r)/10^(7.5),'LineWidth', 2)
hold on
plot([R/fm R/fm], [0, max(u(r)*1.1)/10^(7.5)])


% check normalization
rr = linspace(0, 100000, 1000000) * fm; % 100000 approx. inf
norm = trapz(rr, u(rr).^2) % 0.997583925 approx. 1

%% Calculating Root-Mean-Square Radius
r_dsq = trapz(rr, 0.25*rr.^2.*u(rr).^2); % numerically approximate integral
r_d = sqrt(r_dsq) % 1.92115 fm
