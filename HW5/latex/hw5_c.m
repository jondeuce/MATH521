% Plot triangular meshes and corresponding mass matrix sparsity patterns
% for 'video10.mat' and 'kiwi.mat' datasets

for fname = {'video10', 'kiwi'}
    msh = load(strcat(fname{1},'.mat'));
    Mbar = discretiseLinearElasticity(msh);
    
    figure, subplot(1,2,1);
    pdeplot(msh.P,msh.E,msh.T);
    axis equal
    
    subplot(1,2,2);
    spy(Mbar);
end

% Plot my favourite function uh on the kiwi domain, and compute its L2-norm

u = @(x1,x2) sin(2.*pi.*x1).*cos(6.*pi.*x2);
msh = load('kiwi.mat');
Mbar = discretiseLinearElasticity(msh);

uh = u(msh.P(1,:),msh.P(2,:)).';
L2norm = uh'*Mbar*uh;

figure
pdeplot(msh.P,msh.E,msh.T,'xydata',uh);
title(sprintf('$||u^h||^2_2 = %6f$',L2norm));
axis equal
