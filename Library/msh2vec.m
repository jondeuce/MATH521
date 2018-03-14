function [ u ] = msh2vec( U, msh )
%MSH2VEC Applies lexicographical ordering to re-arrange the rectangular
% array U to a column vector.
% 
% This function takes two input variables:
%   U:   an array of size (msh.N(2) - 1) x (msh.N(1) - 1) corresponding to
%        function values on the interior nodes of the grid only, not the 
%        boundary nodes
%   msh: the output of meshRectangle
% 
% This function returns one output variable:
%   u:   a lexiographically reordered column vector of size
%        (msh.N(1) - 1)(msh.N(2) - 1) x 1
% 
% Notes:
%   The first component of u should contain the value of U that corresponds
%   to the bottom left interior grid point of the rectangular domain, the 
%   second component should correspond to the next grid point to the right 
%   etc. Moving from left to right row by row, the last component of u will
%   be equal to the value of U that corresponds to the interior grid point
%   in the top right corner of the domain.

% Suppose the input array U is given by U = [ 1 2 3 4
%                                             5 6 7 8 ]
% and the final array u should be u = [1;2;3;4;5;6;7;8]. Then, we have that
%              [ 1 5
%   transpose    2 6   reshape  [ 1; 2; 3; 4; 5; 6; 7; 8 ]
%      ==>       3 7     ==>
%                4 8 ]
u = reshape(U.', (msh.N(1)-1)*(msh.N(2)-1), 1);

end