# Imports for math
import numpy
import sympy


# Define the variables and funcions used
m1, m2, L1, L2, t, g = sympy.symbols('m1 m2 L1 L2 t g')

m1 = 1
m2 = 1
L1 = 1
L2 = 1
g = 9.8

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

L = T - V

print(L)