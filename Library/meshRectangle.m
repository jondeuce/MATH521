function [ msh ] = meshRectangle( x, N )
%MESHRECTANGLE Meshes a two-dimensional rectangular domain.
% This function takes two input variables:
%   x:  1 × 4 array
%       Defines the coordinates of the rectangle
%       [x(1),x(2)] x [x(3),x(4)]
%   N:  1 × 2 array
%       Specifies that the domain is to be divided into N(1)/N(2)
%       subintervals in the x1/x2-direction, respectively
% 
% This function returns one output variable:
%   msh (structure with the following fields):
%       X1: (N(2)+1) x (N(1)+1) array containing x1 at each gridpoint
%       X2: (N(2)+1) x (N(1)+1) array containing x2 at each gridpoint
%       N:  a copy of the input variable of the same name
%       h:  1 x 2 array which contains the width of the subintervals
%           in the x1 and x2-direction

% h is the width of each side of the domain divided by the number of
% points on each side:
h  = [x(2)-x(1), x(4)-x(3)]./N;

% X1, X2 are output by the meshgrid function, which essentially just
% reshapes and replicates the (linearly spaced) input row-vectors
% along the appropriate dimensions
x1 = linspace(x(1),x(2),N(1)+1);
x2 = linspace(x(3),x(4),N(2)+1);
[X1,X2] = meshgrid(x1,x2);

% Finally, assemble the output struct msh
msh = struct( ...
    'X1', X1, ... % array of x1 values at each gridpoint
    'X2', X2, ... % array of x2 values at each gridpoint
    'N',  N,  ... % number of subintervals in each direction
    'h',  h   ... % grid spacing in each direction
    );

end

