import sympy

a1, a2, a3, a4 = sympy.symbols('a1, a2, a3, a4')
x1, x2, x3, x4 = sympy.symbols('x1, x2, x3, x4')
y1, y2, y3, y4 = sympy.symbols('y1, y2, y3, y4')
z1, z2, z3, z4 = sympy.symbols('z1, z2, z3, z4')
t1, t2, t3, t4 = sympy.symbols('t1, t2, t3, t4')
eq1 = sympy.Eq(a1 + a2 * x1 + a3 * y1 + a4 * z1, t1)
eq2 = sympy.Eq(a1 + a2 * x2 + a3 * y2 + a4 * z2, t2)
eq3 = sympy.Eq(a1 + a2 * x3 + a3 * y3 + a4 * z3, t3)
eq4 = sympy.Eq(a1 + a2 * x4 + a3 * y4 + a4 * z4, t4)

solution = sympy.solve((eq1, eq2, eq3, eq4), (a1, a2, a3, a4))
alpha_1 = sympy.collect(solution[a1], (t1, t2,t3,t4))
expr = solution[a1].subs([(x1, 1), (x2, 9), (x3, 5), (x4, 8), (y1, 0), (y2, 7), (y3, 6), (y4, 1)])
sympy.pprint(sympy.collect(expr, (t1, t2, t3, t4)))

ak, bk, ck, dk = sympy.symbols('ak, bk, ck, dk')
x, y, z = sympy.symbols('x, y, z')
phi_i_sq = sympy.Eq(bk**2*x**2 + ck**2*y**2 + dk**2*z**2 + 2*ak*bk*x + 2*ak*ck*y + 2*ak*dk*z + 2*bk*ck*x*y + 2*ck*dk*y*z
					+ 2*bk*dk*x*z + ak**2, 0)
#int_phi_i_sq = sympy.integrate(bk**2*x**2 + ck**2*y**2 + dk**2*z**2 + 2*ak*bk*x + 2*ak*ck*y + 2*ak*dk*z + 2*bk*ck*x*y + 2*ck*dk*y*z
#					+ 2*bk*dk*x*z + ak**2, (x, 0,1), (y, 0, 1), (z, 0, 1))
int_phi_i_sq = sympy.integrate(bk, (x, 0,1), (y, 0, 1))
sympy.pprint(int_phi_i_sq)

