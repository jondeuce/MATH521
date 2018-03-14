function [ msh ] = rectMeshFEM( xb, yb, Nx, Ny )
%RECTFEMMESH [ msh ] = rectMeshFEM( xb, yb, Nx, Ny )

nT = 2*Nx*Ny;
nRow = Nx+1;
nCol = Ny+1;
mSiz = [nRow, nCol];

[X,Y] = meshgrid(linspace(xb(1),xb(2),Nx+1),...
                 linspace(yb(1),yb(2),Ny+1));
X = X.';
Y = Y.';
msh.P = [X(:), Y(:)].';

[II,JJ] = meshgrid(1:Nx, 1:Ny);
rowVec = @(x) x(:).';

msh.T = [ rowVec( sub2ind(mSiz, II,   JJ  ) ), rowVec( sub2ind(mSiz, II,   JJ  ) )
          rowVec( sub2ind(mSiz, II+1, JJ  ) ), rowVec( sub2ind(mSiz, II+1, JJ+1) )
          rowVec( sub2ind(mSiz, II+1, JJ+1) ), rowVec( sub2ind(mSiz, II,   JJ+1) )
          ones(1,nT) ];

msh.E  = [ ...
    [ sub2ind(mSiz, (1:nRow-1), ones(1,nRow-1))
      sub2ind(mSiz, (2:nRow),   ones(1,nRow-1)) ], ...
    [ sub2ind(mSiz, nRow*ones(1,nCol-1), (1:nCol-1) )
      sub2ind(mSiz, nRow*ones(1,nCol-1), (2:nCol)) ], ...
    [ sub2ind(mSiz, (nRow:-1:2),   nCol*ones(1,nRow-1))
      sub2ind(mSiz, (nRow-1:-1:1), nCol*ones(1,nRow-1)) ], ...
    [ sub2ind(mSiz, ones(1,nCol-1), (nCol:-1:2))
      sub2ind(mSiz, ones(1,nCol-1), (nCol-1:-1:1)) ] ...
    ];

end

