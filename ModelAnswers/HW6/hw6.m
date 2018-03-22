% hw6.m
close all; clc;

% problem data
a = 1e-1;
c = 1;
f = @(x1,x2) 7.*exp(-7.*(x1-1.4).^2-7.*(x2-1).^2) + 10*exp(-10.*(x1-.5).^2-10.*(x2-1.2).^2);
g = @(x1,x2) 5e-1.*exp(-5e-1.*(x1-1.4).^2-5e-1.*(x2-1).^2);
GammaD = @(x1,x2) true(size(x1));
% GammaD = @(x1,x2) false(size(x1));
% GammaD = @(x1,x2) x1 < 1 | x2 > .6;
plotting = input('Shall I use TRIMESH / TRISURF (1) or PDEMESH / PDEPLOT (2)? ');
msh = load('kiwi.mat','P','E','T');

[A,b,uD,Kbar,Mbar,Pf,PD] = discretiseLinearElasticity(c,a,f,g,GammaD,msh);

% solve linear system and add boundary values
ubar = Pf'*(A\b) + PD'*uD;

% compute norms
L2 = sqrt(ubar'*Mbar*ubar);
H1 = sqrt(ubar'*Mbar*ubar + ubar'*Kbar*ubar);
en = sqrt(c*(ubar'*Kbar*ubar) + a*(ubar'*Mbar*ubar));
fprintf('L²-norm = %3.2f\nH¹-norm = %3.2f\n B-norm = %3.2f\n',L2,H1,en);

% plot the mesh
figure;
switch plotting
    case 1
        % use trimesh
        trimesh(msh.T(1:3,:)',msh.P(1,:)',msh.P(2,:)');
    otherwise
        % use pdemesh
        pdemesh(msh.P,msh.E,msh.T);
end
axis equal
xlabel('{\itx}_1');
ylabel('{\itx}_2');

% plot the solution
figure;
switch plotting
    case 1
        % use trisurf
        trisurf(msh.T(1:3,:)',msh.P(1,:)',msh.P(2,:)',ubar);
        colormap summer
    otherwise
        % use pdeplot
        pdeplot(msh.P,msh.E,msh.T,'xydata',ubar,'zdata',ubar,'mesh','on','colormap','summer');
end
axis equal
xlabel('{\itx}_1');
ylabel('{\itx}_2');
zlabel('{\itu^h}({\itx}_1,{\itx}_2)');
title('Elongation');

% plot the force
figure;
switch plotting
    case 1
        % use trisurf
        fp = f(msh.P(1,:),msh.P(2,:))'; % function values on vertices
        trisurf(msh.T(1:3,:)',msh.P(1,:)',msh.P(2,:)',fp);
        colormap summer
    otherwise
        % use pdeplot
        xm = (msh.P(:,msh.T(1,:)) + msh.P(:,msh.T(2,:)) + msh.P(:,msh.T(3,:)))./3; % triangle midpoints
        fm = f(xm(1,:),xm(2,:)); % function values on triangle midpoints
        % since we're using the midpoint rule to evaluate the right-hand side, I'm also plotting f as a piecewise constant function
        pdeplot(msh.P,msh.E,msh.T,'xydata',fm,'zdata',fm,'mesh','on','colormap','summer','xystyle','flat','zstyle','discontinuous');
end
axis equal
xlabel('{\itx}_1');
ylabel('{\itx}_2');
zlabel('{\itf}({\itx}_1,{\itx}_2)');
title('Force');
