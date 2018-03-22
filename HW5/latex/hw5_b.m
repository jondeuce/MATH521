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
