%% create variables:
syms x1 x2 x3 y1 y2 y3 z1 z2 z3 x y z real
planeEqn = det([  x-x1,  y-y1,  z-z1
                 x2-x1, y2-y1, z2-z1
                 x3-x1, y3-y1, z3-z1 ]);
PlaneEqn = matlabFunction(planeEqn);

%% plane with (x1,y1,z1)=(0,0,1), (x2,y2,z2)=(x2,0,0), and (x3,y3,z3)=(x3,y3,0)
X1 = 0; Y1 = 0; Y2 = 0; Z1 = 1; Z2 = 0; Z3 = 0;
unitPlane = solve(PlaneEqn(x, X1, x2, x3,...
                           y, Y1, Y2, y3,...
                           z, Z1, Z2, Z3), z);
UnitPlane = matlabFunction(unitPlane);

X = (X1+x2+x3)/3;
Y = (Y1+Y2+y3)/3;

fprintf('Centroid: ');
disp(simplify(UnitPlane(X,x2,x3,Y,y3))); % = 1/3

fprintf('Middle of edge: ');
disp(simplify(UnitPlane(x2/2,x2,x3,0,y3))); % = 1/2

%% General plane

generalPlane = solve(planeEqn,z);
GeneralPlane = matlabFunction(generalPlane);

X = (x1+x2+x3)/3;
Y = (y1+y2+y3)/3;

fprintf('Centroid: ');
disp(simplify(GeneralPlane(X,x1,x2,x3,Y,y1,y2,y3,Z1,Z2,Z3))); % = 1/3

X = (x1+x2)/2;
Y = (y1+y2)/2;
fprintf('Middle of edge #1: ');
disp(simplify(GeneralPlane(X,x1,x2,x3,Y,y1,y2,y3,Z1,Z2,Z3))); % = 1/2

X = (x2+x3)/2;
Y = (y2+y3)/2;
fprintf('Middle of edge #2: ');
disp(simplify(GeneralPlane(X,x1,x2,x3,Y,y1,y2,y3,Z1,Z2,Z3))); % = 1/2

X = (x3+x1)/2;
Y = (y3+y1)/2;
fprintf('Middle of edge #3: ');
disp(simplify(GeneralPlane(X,x1,x2,x3,Y,y1,y2,y3,Z1,Z2,Z3))); % = 1/2

%% Triangle area
T = 1/2 * simplify(norm(cross([x1,y1,0],[x2,y2,0]))); % T = 1/2*|x1*y2 - x2*y1|
fprintf('Area = ');
disp(T);

