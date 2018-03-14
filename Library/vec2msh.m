function [ U ] = vec2msh( u, msh )
%MSH2VEC Rearranges the lexicographically re-ordered column vector u to the
%rectangular array U of size (msh.N(2) - 1) x (msh.N(1) - 1).
% 
% This function takes two input variables:
%   u:   a lexiographically reordered column vector of size
%        (msh.N(1) - 1)*(msh.N(2) - 1) x 1
%   msh: the output of meshRectangle
% 
% This function returns one output variable:
%   U:   an array of size (msh.N(2) - 1) x (msh.N(1) - 1) corresponding to
%        function values on the interior nodes of the grid only, not the 
%        boundary nodes
% 
% Notes:
%   The first component of u should contain the value of U that corresponds
%   to the bottom left interior grid point of the rectangular domain, the 
%   second component should correspond to the next grid point to the right 
%   etc. Moving from left to right row by row, the last component of u will
%   be equal to the value of U that corresponds to the interior grid point
%   in the top right corner of the domain.

% Suppose our u is the column vector u = [1;2;3;4;5;6;7;8], and we desire
% our final array to be U = [ 1 2 3 4
%                             5 6 7 8 ].
% Then, we have that:
%            [ 1 5
%   reshape    2 6  transpose  [ 1 2 3 4
%     ==>      3 7     ==>       5 6 7 8 ]
%              4 8 ]
U = reshape(u, msh.N(1)-1, msh.N(2)-1).';

end