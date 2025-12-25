# Imports for math
import numpy
import sympy
from scipy.integrate import odeint
import Functions

# Define the variables and funcions used
m1, m2, L1, L2, t, g = sympy.symbols('m1 m2 L1 L2 t g')

theta1, theta2 = sympy.symbols('theta1, theta2', cls=sympy.Function)
theta1 = theta1(t)
theta2 = theta2(t)

x1 = L1 * sympy.sin(theta1)
y1 = - L1 * sympy.cos(theta1)
x2 = x1 + L2 * sympy.sin(theta2)
y2 = y1 - L2 * sympy.cos(theta2)


# Derivatives
theta1_dot = sympy.diff(theta1, t)
theta2_dot = sympy.diff(theta2, t)
theta1_ddot = sympy.diff(theta1_dot, t)
theta2_ddot = sympy.diff(theta2_dot, t)

x1_dot = sympy.diff(x1, t)
x2_dot = sympy.diff(x2, t)
y1_dot = sympy.diff(y1, t)
y2_dot = sympy.diff(y2, t)

# Lagrangian
T1 = 1/2 * m1 * (x1_dot**2 + y1_dot**2)
T2 = 1/2 * m2 * (x2_dot**2 + y2_dot**2)
T = T1 + T2

V1 = m1 * g * y1
V2 = m2 * g * y2
V = V1 + V2

L = sympy.simplify(T - V)

# Lagrange equations of motion
LE1 = sympy.simplify(theta1_dot - theta1_ddot)
LE2 = sympy.simplify(theta2_dot - theta2_ddot)

sols = sympy.solve([LE1, LE2], [theta1_ddot, theta2_ddot])
sols[theta1_ddot] = sympy.simplify(sols[theta1_ddot])
sols[theta2_ddot] = sympy.simplify(sols[theta2_ddot])

# Turning into first order ODEs
dz1dt = sympy.lambdify((m1, m2, L1, L2, t, g, theta1, theta2, theta1_dot, theta2_dot), 
                       sols[theta1_ddot])
dtheta1dt = sympy.lambdify(theta1_dot, theta1_dot)

dz2dt = sympy.lambdify((m1, m2, L1, L2, t, g, theta1, theta2, theta1_dot, theta2_dot), 
                       sols[theta2_ddot])
dtheta2dt = sympy.lambdify(theta2_dot, theta2_dot)

# Solving ODEs
m1 = 1
m2 = 1
L1 = 1
L2 = 1
t = numpy.linspace(0, 30, 1000)
g = 9.8
dSdt = Functions.dSdt

ans = odeint(dSdt, y0=[1, 1, 1, 1], t = t, agrs = (m1, m2, L1, L2, g))