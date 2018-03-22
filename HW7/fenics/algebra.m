%% vars
syms x y real

%% func
g = x*(1-x) + y*(1-y);
D = 1;
a = [1,1];
r = 1;

u = g;
Lap = @(u) diff(u,x,2) + diff(u,y,2);
Div = @(v) diff(v(1),x,1) + diff(v(2),y,1);
f = -D * Lap(u) + Div(a * u) + r * u