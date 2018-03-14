function [ ] = q1( )
%Q1 [d^2/dx1^2 + d^2/dx1dx2 + x2 d^2/dx2^2]u + 1/3u^3 = 0

%% Q1A
fprintf('This PDE is:\n');
fprintf('\t-> semi-linear, as coeffs of principle part depend only on x1,x2\n');
fprintf('\t-> homogeneous, as coeffs of principle part depend only on x1,x2\n');
fprintf('\t-> of 2nd order, there are second derivates of u w.r.t x1,x2\n');
fprintf('\t-> in 2 variables, as there are two coordinates x1,x2\n');
fprintf('\n');

%% Q1B
% We have a_11 = 1, a_12 = 1/2, and a_22 = x2, thus the discriminant is:
%     D = (1/2)^2 - 1*x2
%       = 1/4 - x2
fprintf('This PDE is:\n');
fprintf('\t-> elliptic,   if x2 > 1/4\n');
fprintf('\t-> parabolic,  if x2 = 1/4\n');
fprintf('\t-> hyperbolic, if x2 < 1/4\n');
fprintf('\n');

end

