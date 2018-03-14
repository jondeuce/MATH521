function Mbar = assembleMass(msh)
%ASSEMBLEMASS Assembles the complete mass matrix including all boundary
%points, discretised with linear finite elements

% need all pairs points, including pairing with themselves:
pointRows = [1,2,3];
pointsAndNeighboursRows = [1,2,3, 2,3,1, 3,1,2];

% Assembly II and JJ arrays such that (II(idx), JJ(idx)) pairs contain all
% pairs of triangle points with themselves and their neighbours
%   NOTE: There will certainly be repeated points, as triangles share
%         vertices. However this is accounted for in the sparse constructor
%         as it adds all repeated indices together, i.e. all contributions
%         from all element mass matrices
II = msh.T(pointsAndNeighboursRows, :);
JJ = repmat( msh.T(pointRows, :), 3, 1);

% First row will be pairs of points with themselves, so value is 2/12*|T|
% 2nd/3rd rows are pairs of points with other points, so value is 1/12*|T|
S  = repmat( [ (2/12) * msh.A
               (1/12) * msh.A
               (1/12) * msh.A ], 3, 1);

% complete mass matrix
Mbar = sparse(II,JJ,S,msh.np,msh.np);

end
