# Imports for math
import numpy as np
import sympy as smp
from scipy.integrate import odeint
#import Functions
import matplotlib.pyplot

# Define the variables and funcions used
t, m1, m2, L1, L2, g = smp.symbols('t m1 m2 L1 L2 g')

theta1, theta2 = smp.symbols('theta1, theta2', cls=smp.Function)
theta1 = theta1(t)
theta2 = theta2(t)

x1 = L1 * smp.sin(theta1)
y1 = - L1 * smp.cos(theta1)
x2 = x1 + L2 * smp.sin(theta2)
y2 = y1 - L2 * smp.cos(theta2)


# Derivatives
theta1_dot = smp.diff(theta1, t)
theta2_dot = smp.diff(theta2, t)
theta1_ddot = smp.diff(theta1_dot, t)
theta2_ddot = smp.diff(theta2_dot, t)

x1_dot = smp.diff(x1, t)
x2_dot = smp.diff(x2, t)
y1_dot = smp.diff(y1, t)
y2_dot = smp.diff(y2, t)

# Lagrangian
T1 = 1/2 * m1 * (x1_dot**2 + y1_dot**2)
T2 = 1/2 * m2 * (x2_dot**2 + y2_dot**2)
T = T1 + T2

V1 = m1 * g * y1
V2 = m2 * g * y2
V = V1 + V2

L = T - V

# Lagrange equations of motion
#LE1 = smp.simplify(theta1_dot - theta1_ddot)
#LE2 = smp.simplify(theta2_dot - theta2_ddot)
LE1 = smp.diff(L, theta1) - smp.diff(smp.diff(L, theta1_dot), t)#.splimify()
LE2 = smp.diff(L, theta2) - smp.diff(smp.diff(L, theta2_dot), t)#.simplify()

sols = smp.solve([LE1, LE2], [theta1_ddot, theta2_ddot])
#sols[theta1_ddot] = smp.simplify(sols[theta1_ddot])
#sols[theta2_ddot] = smp.simplify(sols[theta2_ddot])

# Turning into first order ODEs
dz1dt = smp.lambdify((t, m1, m2, L1, L2, g, theta1, theta2, theta1_dot, theta2_dot), 
                       sols[theta1_ddot])
dtheta1dt = smp.lambdify(theta1_dot, theta1_dot)

dz2dt = smp.lambdify((t, m1, m2, L1, L2, g, theta1, theta2, theta1_dot, theta2_dot), 
                       sols[theta2_ddot])
dtheta2dt = smp.lambdify(theta2_dot, theta2_dot)

# Define state vector
def dSdt(S, t, m1, m2, L1, L2, g):
    theta1, z1, theta2, z2 = S
    return[
        dtheta1dt(z1),
        dz1dt(t, m1, m2, L1, L2, g, theta1, theta2, z1, z2),
        dtheta2dt(z2),
        dz2dt(t, m1, m2, L1, L2, g, theta1, theta2, z1, z2)
    ]


# Solving ODEs
m1 = 2
m2 = 1
L1 = 2
L2 = 1
t = np.linspace(0, 40, 1001)
g = 9.81

ans = odeint(dSdt, y0=[1, -3, -1, 5], t=t, args=(m1, m2, L1, L2, g))

# Functions theta1 and theta2 in terms of time
theta1 = ans.T[0]
theta2 = ans.T[2]