%% create vars
syms M u1 u0 v1 v0 K theta dt c

%% initial eqns
LHS1 = M*u1 - theta*dt*M*v1;
RHS1 = M*u0 + (1-theta)*dt*M*v0;
LHS2 = theta*dt*c^2*K*u1 + M*v1;
RHS2 = -(1-theta)*dt*c^2*K*u0 + M*v0;

%% first set of new eqns
simp = @(expr) collect(collect(collect(collect(simplify(expand(expr)),u0),v0),u1),v1);

LHS3 = simp( LHS1 + LHS2*(theta*dt) )
RHS3 = simp( RHS1 + RHS2*(theta*dt) )

%% second set of new eqns
LHS4 = simp( LHS2 - theta*dt*c^2*K*u1 )
RHS4 = simp( RHS2 - theta*dt*c^2*K*u1 )

