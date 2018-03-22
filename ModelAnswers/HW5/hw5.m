% hw5.m
close all; clc;

plotting = input('Shall I use TRIMESH / TRISURF (1) or PDEMESH / PDEPLOT (2)? ');

%% Question 2(b)

data = {'video10','kiwi'};

for k = 1:length(data)
    msh = load([data{k} '.mat'],'P','E','T');
    M = discretiseLinearElasticity(msh);
    figure;
    subplot(1,2,1);
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
    title(data{k});
    subplot(1,2,2); spy(M);
end

%% Question 2(c)

% favourite function of the day
u = @(x1,x2) exp(-5e-1*(x1-1.4).^2 - 5e-1*(x2-1).^2);
uh = u(msh.P(1,:),msh.P(2,:))';

fprintf('L²-norm = %3.2f\n',sqrt(uh'*M*uh));

figure;
switch plotting
    case 1
        % use trisurf
        trisurf(msh.T(1:3,:)',msh.P(1,:)',msh.P(2,:)',uh);
        colormap summer
    otherwise
        % use pdeplot
        pdeplot(msh.P,msh.E,msh.T,'xydata',uh,'zdata',uh,'mesh','on','colormap','summer');
end
axis equal
xlabel('{\itx}_1');
ylabel('{\itx}_2');
zlabel('{\itu^h}({\itx}_1,{\itx}_2)');
