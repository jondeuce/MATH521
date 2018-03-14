close all force

Ncrit = 128; % Ncrit = a/2D
D = 0.1;
a = 2*D*Ncrit;

pmin = 2;
pmax = 2*round(log2(Ncrit));
Nrange = 2.^(pmin:pmax);

err_CD_last = 0;
err_UD_last = 0;
fprintf('N       h           err_CD      rat_CD      min_CD      err_UD      rat_UD      min_UD\n');
for N = Nrange
    [u_CD,u_UD,u_ex,ubar_ex,xgrid] = adv_diff(a,D,N);
    %advection_diffusion(a,D,N)
    
    err_f = @(u,u_ex) sqrt(trapz(xgrid,(u-u_ex).^2));
    err_CD = err_f(u_CD,u_ex);
    %err_UD = err_f(u_UD,u_ex);
    err_UD = err_f(u_UD,ubar_ex);
    fprintf('%5d, %10.3e, %10.3e, %10.3e, %10.3e, %10.3e, %10.3e, %10.3e\n', ...
        N, 1/N, err_CD, err_CD_last/err_CD, min(u_CD), err_UD, err_UD_last/err_UD, min(u_UD));
    err_CD_last = err_CD;
    err_UD_last = err_UD;
end
