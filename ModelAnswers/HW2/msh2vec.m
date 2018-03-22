function u = msh2vec(U,msh)
%MSH2VEC re-arranges a rectangular array of function values to a column
%vector in lexicographical order
%   Input:
%       U: an array of size (msh.N(2) - 1) × (msh.N(1) - 1)
%       msh: a structure as defined by meshRectangle
%   Output:
%       u: an array of size (msh.N(1) - 1)(msh.N(2) - 1) × 1

u = reshape(U',prod(msh.N-1),1);