%% vars
syms h a D d positive
syms x real
syms u(x)

%% funs
Fu = taylor(u(x+h),h,'order',5);
Bu = taylor(u(x-h),h,'order',5);
du = diff(u,x);
d2u = diff(u,x,2);

Ah_cd = simplify((Fu-Bu)/(2*h));
Ah_ud = simplify((u-Bu)/h);
Dh = simplify((Fu-2*u+Bu)/h^2);

%% Testing
CD_err = collect(simplify( (a*du - D*d2u) - (a*Ah_cd - D*Dh) ),h)
UD_err = collect(simplify( (a*du - D*d2u) - (a*Ah_ud - D*Dh) ),h)

d = a*h/2;
UDbar_err = collect(simplify( (a*du - (D+d)*d2u) - (a*Ah_ud - D*Dh) ),h)
