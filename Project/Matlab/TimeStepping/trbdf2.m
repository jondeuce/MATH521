function [t,u] = trbdf2(args,tspan,u0)
% [t,u] = trbdf2({A,M,dt},tspan,u0)
%   (M + (1 - 1/rt2)*A*dt)*ua == M*u0 - ((1 - 1/rt2)*dt)*A*u0
%   (M + (1 - 1/rt2)*A*dt)*up == ((1+rt2)/2)*M*ua + ((1-rt2)/2)*M*u0

[A,M,dt] = deal(args{:});

c1 = (1 - 1/sqrt(2));
c2 = (1 + sqrt(2))/2;
c3 = (1 - sqrt(2))/2;

t = (tspan(1):dt:tspan(2)).';
u = zeros(length(t),length(u0));
u(1,:) = u0.';

B = M + (c1*dt)*A;

for ii = 2:length(t)
    L1 = M*u0 - (c1*dt)*(A*u0);
    ua = B\L1;
    
    L2 = M*(c2*ua + c3*u0);
    up = B\L2;
    
    u(ii,:) = up.';
    u0 = up;
end

end
