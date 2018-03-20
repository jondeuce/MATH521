function U = vec2msh(u,msh)
%VEC2MSH re-arranges a column vector of function values back to a
% rectangular array; inverse of MSH2VEC
%   Input:
%       u: an array of size (msh.N(1) - 1)(msh.N(2) - 1) × 1
%       msh: a structure as defined by meshRectangle
%   Output:
%       U: an array of size (msh.N(2) - 1) × (msh.N(1) - 1)

U = reshape(u,msh.N-1)';