function msh = meshRectangle(x,N)
%MESHRECTANGLE meshes the domain [x(1), x(2)] x [x(3), x(4)] with N(1)
%subintervals in x1-direction and N(2) subintervals in x2-direction.
%   Input:
%       x: a 1 x 4 array
%       N: a 1 x 2 array
%   Output:
%       msh: a structure with fields
%           X1 and X2: arrays of size (N(2) + 1) × (N(1) + 1) that contain
%                      the x1 or x2 components, respectively, of the grid
%                      points
%           N:         a copy of the input variable N
%           h:         a 1 x 2 array with the grid spacing in x1/x2
%                      direction


[msh.X1,msh.X2] = meshgrid(linspace(x(1),x(2),N(1)+1),linspace(x(3),x(4),N(2)+1));
msh.N = N;
msh.h = [x(2)-x(1), x(4)-x(3)]./N;