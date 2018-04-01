% Check that a block matrix of the form
% 	[   A, +/-W]
%   [-/+W,    A]
% (where A is positive definite and W is symmetric) is positive definite

% random symm pos def A
n = 5;
A = randn(n,n); % generate a random n x n matrix
A = 0.5*(A+A')/maxabs(A);
A = A + n*eye(n);

% random symm W
W = randn(n);
W = W + W';

B = [A, W; -W, A];
d = eig(B)