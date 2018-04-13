%% vars
syms ua up u0 dt A
simp = @(x) collect(collect(collect(collect(collect(expand(x),dt),A),u0),ua),up);
checkeq = @(x,y) simplify(expand(x - y)) == sym(0);

rt2 = sqrt(sym(2));
a = 2 - rt2;

%% algebra: TRBDF2 method with f(t,u) = -A*u
eqn1 = simp(-ua + u0 + a*dt*(-A*ua - A*u0)/2)
equiv_eqn1 = -(1 + (1 - 1/rt2)*A*dt)*ua + (1 - A*dt*(1 - 1/rt2))*u0
checkeq(eqn1, equiv_eqn1)
% (I + (1 - 1/rt2)*A*dt)*ua == (I - A*dt*(1 - 1/rt2))*u0

eqn2 = simp(-(2-a)*up + ua/a - (1-a)^2/a*u0 + (1-a)*dt*(-A*up));
eqn2 = simp((1/rt2)*eqn2)
equiv_eqn2 = -(1 + A*dt*(1 - 1/rt2))*up + (1/2)*(1-rt2)*u0 + (1/2)*(1+rt2)*ua
checkeq(eqn2, equiv_eqn2)
% (I + (1 - 1/rt2)*A*dt)*up == 1/2*(1-rt2)*u0 + 1/2*(1+rt2)*ua

%% finally, let A -> M^-1 * A, and multiply through by M:
%   (M + (1 - 1/rt2)*A*dt)*ua == M*u0 - ((1 - 1/rt2)*dt)*A*u0
%   (M + (1 - 1/rt2)*A*dt)*up == ((1+rt2)/2)*M*ua + ((1-rt2)/2)*M*u0
