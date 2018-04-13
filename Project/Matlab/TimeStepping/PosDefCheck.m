% Check that a block matrix of the form
% 	[   A, +/-W]
%   [-/+W,    A]
% (where A is positive definite and W is symmetric) is positive definite

% random symm pos def A
n = 5;
A = randSPDmat(n);

% random symm W
W = randn(n);
W = W + W';

B = [A, W; -W, A];
d = eig(B)
