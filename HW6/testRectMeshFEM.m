%% symbolic functions
syms x y real
PI = sym('pi');

%% symbolic functions
f = cos(2*PI*x).*sin(4*PI*y) + exp(-(x+y).^2);
dfdx = diff(f,x);
dfdy = diff(f,y);

%% function handles
F = matlabFunction(f);
dFdx = matlabFunction(dfdx);
dFdy = matlabFunction(dfdy);
dF = @(x,y) cat(3,dFdx(x,y),dFdy(x,y));

%% discretiseLinearElasticity args
c = 1;
a = 1;
f = @(x1,x2) bsxfun(@times, sin(x1), cos(x2));
g = @(x1,x2) zeros(size(x1));
GammaD = @(x1,x2) true(size(x1));

%% setup test params
NX = 2.^(1:9); NY = round(3*(NX/2) + 5);
xb = [-rand,rand];
yb = [-rand,rand];

int2args = {xb(1),xb(2),yb(1),yb(2),'abstol',1e-14,'reltol',1e-14};
integrate = @(fcn) integral2( fcn, int2args{:} );

L2normsq = integrate(@(x,y) F(x,y).^2 );
L2gradsq = integrate(@(x,y) sum(dF(x,y).^2,3) );

%% run test loop
fprintf('\n[ Nx, Ny]  normsq     normsq_h   e_normsq   ratio      gradsq     gradsq_h   e_gradsq   ratio\n');
eh_last_normsq = 0;
eh_last_gradsq = 0;
for ii = 1:numel(NX)
    Nx = NX(ii);
    Ny = NY(ii);
    msh = rectMeshFEM( xb, yb, Nx, Ny );
    [A,b,uD,Kbar,Mbar,Pf,PD] = discretiseLinearElasticity(c,a,f,g,GammaD,msh);
    
    uh = F(msh.P(1,:),msh.P(2,:)).';
    
    L2normsq_h = uh'*Mbar*uh;
    eh_L2normsq = abs(L2normsq-L2normsq_h);
    
    L2gradsq_h = uh'*Kbar*uh;
    eh_L2gradsq = abs(L2gradsq-L2gradsq_h);
    
    
    fprintf('[%3d,%3d]  %8.3e  %8.3e  %8.3e  %8.3e  %8.3e  %8.3e  %8.3e  %8.3e\n', Nx, Ny, ...
        L2normsq, L2normsq_h, eh_L2normsq, eh_last_normsq/eh_L2normsq, ...
        L2gradsq, L2gradsq_h, eh_L2gradsq, eh_last_gradsq/eh_L2gradsq );
    
    eh_last_normsq = eh_L2normsq;
    eh_last_gradsq = eh_L2gradsq;
end
