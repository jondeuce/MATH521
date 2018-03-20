function [A,b,uD,Kbar,Mbar,Pf,PD] = discretiseLinearElasticity(c,a,f,g,GammaD,msh)
%DISCRETISELINEARELASTICITY(c,a,f,g,GammaD,msh) Applies linear finite
% elements on the triangulation defined in the structure msh to discretise
% the equation
%   -c*Lap(u) + a*u = f
% with Dirichlet boundary conditions u = g on GammaD.
%   Input:
%       c: a positive real number
%       a: a positive real number
%       f: a function handle for the inhomogeneity f(x1,x2)
%       g: a function handle for the boundary values g(x1,x2)
%       GammaD: a function handle to a function of x1 and x2, which returns
%       true, if Dirichlet conditions are imposed at the boundary point
%       (x1,x2) and false otherwise
%       msh: a mesh structure with fields P, E, T
%   Output:
%       A: a sparse array corresponding to the discrete elliptic operator
%       b: a column vector for the discretised right hand side and boundary
%       terms
%       uD: a column vector with the prescribed values of the solution on
%       the Dirichlet boundary
%       Kbar: the sparse stiffness matrix, including all boundary points
%       Mbar: the sparse mass matrix, including all boundary points
%       Pf: a sparse matrix that projects onto all free nodes
%       PD: a sparse matrix that projects onto all Dirichlet nodes

% -------------------------------------------------------------------------
% COMPUTE EXTRA MESH DATA
% -------------------------------------------------------------------------

% edge vectors
msh.D32 = msh.P(:,msh.T(3,:)) - msh.P(:,msh.T(2,:));
msh.D13 = msh.P(:,msh.T(1,:)) - msh.P(:,msh.T(3,:));
msh.D21 = msh.P(:,msh.T(2,:)) - msh.P(:,msh.T(1,:));

% row vector of triangle areas = [|T_1|, |T_2|, |T_3|, ..., |T_nt|]
msh.A = (msh.D13(1,:).*msh.D21(2,:) - msh.D21(1,:).*msh.D13(2,:))./2;

% number of points
msh.np = size(msh.P,2); % includes boundary points

% number of triangles
msh.nt = size(msh.T,2);

% -------------------------------------------------------------------------
% ASSEMBLE THE DISCRETE SYSTEM
% -------------------------------------------------------------------------

% complete system including all boundary points
Kbar = assembleStiffness(msh);
Mbar = assembleMass(msh);
fbar = assembleLoad(f,msh);

% -------------------------------------------------------------------------
% ELIMINATE DIRICHLET BOUNDARY CONDITIONS
% -------------------------------------------------------------------------

[Pf,PD,uD] = eliminateDirichletBC(g,GammaD,msh);

% contributions from points with prescribed Dirichlet data
kD = Pf*Kbar*PD'*uD;
mD = Pf*fbar - kD;

% reduced system for the free points only
K = Pf*Kbar*Pf';
M = Pf*Mbar*Pf';
A = c*K + a*M;
b = kD + mD;

end

function Mbar = assembleMass(msh)
%ASSEMBLEMASS Assembles the complete mass matrix including all boundary
% points, discretised with linear finite elements

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
mij = repmat( [ (2/12) * msh.A
                (1/12) * msh.A
                (1/12) * msh.A ], 3, 1);

% complete mass matrix
Mbar = sparse(II,JJ,mij,msh.np,msh.np);

end

function Kbar = assembleStiffness(msh)
%ASSEMBLESTIFFNESS Assembles the complete stiffness matrix including all
% boundary points, discretised with linear finite elements

% arrays of row/column indices: [9 x nT] arrays (note semicolons)
II = [msh.T(1,:); msh.T(1,:); msh.T(1,:);
      msh.T(2,:); msh.T(2,:); msh.T(2,:);
      msh.T(3,:); msh.T(3,:); msh.T(3,:)]; % row indices

JJ = [msh.T(1,:); msh.T(2,:); msh.T(3,:);
      msh.T(1,:); msh.T(2,:); msh.T(3,:);
      msh.T(1,:); msh.T(2,:); msh.T(3,:)]; % column indices

% entries of the element stiffness matrices (before division by areas):
% [9 x nT] arrays (note semicolons)
kij = [ dot(msh.D32,msh.D32,1); dot(msh.D32,msh.D13,1); dot(msh.D32,msh.D21,1);
        dot(msh.D13,msh.D32,1); dot(msh.D13,msh.D13,1); dot(msh.D13,msh.D21,1);
        dot(msh.D21,msh.D32,1); dot(msh.D21,msh.D13,1); dot(msh.D21,msh.D21) ];

% computationally efficient division by 4*|T|
kij = bsxfun( @rdivide, kij, 4*msh.A );

% complete stiffness matrix
Kbar = sparse(II,JJ,kij,msh.np,msh.np);

end

function qf = assembleLoad(f,msh)
%ASSEMBLELOAD Assembles the complete load vector including all boundary
%points, discretised with linear finite elements

% triangle midpoints
xm = (msh.P(:,msh.T(1,:)) + msh.P(:,msh.T(2,:)) + msh.P(:,msh.T(3,:)))./3;
fm = f(xm(1,:).',xm(2,:).'); % f(xm)

% quadrature operator (midpoint rule) on one element
II  = msh.T(1:3,:); % row indices
JJ  = repmat(1:msh.nt,3,1); % column indices
qij = repmat(msh.A./3,3,1); % midpoint rule on each triangle

% full quadrature operator
Q = sparse(II,JJ,qij,msh.np,msh.nt);

% complete load vector
qf = Q*fm;

end

function [Pf,PD,uD] = eliminateDirichletBC(g,GammaD,msh)
%ELIMINATEDIRICHLETBC Calculates the vector uD = g(x1,x2) for all boundary
%points (x1,x2) on the Dirichlet section GammaD of the boundary. Also
%assembles the projection matrices Pf and PD.

% indices of all boundary points
indGamma = unique(msh.E(1:2,:));

% coordinates of all boundary points
Gamma = msh.P(:,indGamma);

% points on that part of the boundary where Dirichlet conditions are
% imposed
indD = indGamma(GammaD(Gamma(1,:),Gamma(2,:))); % indices of Dirichlet points
nD = length(indD); % number of Dirichlet points

% all other points of the mesh
indf = setdiff(1:msh.np,indD); % indices of free points
nf = msh.np - nD; % number of free points

% projection onto the Dirichlet points
PD = sparse(1:nD,indD,ones(1,nD),nD,msh.np);

% projection onto the free points
Pf = sparse(1:nf,indf,ones(1,nf),nf,msh.np);

% boundary values of u on the Dirichlet points
uD = g(msh.P(1,indD).',msh.P(2,indD).');

end
