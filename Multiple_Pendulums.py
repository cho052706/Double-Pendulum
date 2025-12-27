import numpy as np
import sympy as smp
#import matplotlib.pyplot as plt
#import matplotlib.animation
#from matplotlib.lines import Line2D
from scipy.integrate import odeint

t, L1, L2 = smp.symbols('t L1 L2')

# Making a pendulum
class DoublePendulum():
    def __init__(self, y0):
        self.y0 = y0

    def pend_ODE_solver (self, t=t):
        t, m1, m2, L1, L2, g = smp.symbols('t m1 m2 L1 L2 g')

        theta1, theta2 = smp.symbols('theta1, theta2', cls=smp.Function)
        theta1 = theta1(t)
        theta2 = theta2(t)

        x1 = L1 * smp.sin(theta1)
        y1 = - L1 * smp.cos(theta1)
        x2 = x1 + L2 * smp.sin(theta2)
        y2 = y1 - L2 * smp.cos(theta2)

        ## Derivatives ##
        theta1_dot = smp.diff(theta1, t)          # Angular velocities
        theta2_dot = smp.diff(theta2, t)
        theta1_ddot = smp.diff(theta1_dot, t)     # Angular acceleration
        theta2_ddot = smp.diff(theta2_dot, t)

        x1_dot = smp.diff(x1, t)                  # Translational velocities
        x2_dot = smp.diff(x2, t)
        y1_dot = smp.diff(y1, t)
        y2_dot = smp.diff(y2, t)

        ## Lagrangian ##
        T1 = 1/2 * m1 * (x1_dot**2 + y1_dot**2)   # Kinetic energy 
        T2 = 1/2 * m2 * (x2_dot**2 + y2_dot**2)
        T = T1 + T2

        V1 = m1 * g * y1                          # Potencial energy
        V2 = m2 * g * y2
        V = V1 + V2

        L = T - V                                 # Lagrangian

        LE1 = smp.diff(L, theta1) - smp.diff(smp.diff(L, theta1_dot), t)
        LE2 = smp.diff(L, theta2) - smp.diff(smp.diff(L, theta2_dot), t)

        sols = smp.solve([LE1, LE2], [theta1_ddot, theta2_ddot])

        ## Convert to 1st-order ODEs ##
        dz1dt = smp.lambdify((t, m1, m2, L1, L2, g, theta1, theta2, theta1_dot, theta2_dot), 
                            sols[theta1_ddot]) # Angular acceleration as 1st-order
        dtheta1dt = smp.lambdify(theta1_dot, theta1_dot)

        dz2dt = smp.lambdify((t, m1, m2, L1, L2, g, theta1, theta2, theta1_dot, theta2_dot), 
                            sols[theta2_ddot]) # Angular acceleration as 1st-order
        dtheta2dt = smp.lambdify(theta2_dot, theta2_dot)

        ### Setting values ###
        m1 = 1
        m2 = 1
        L1 = 1                                    # (ax) might be modified for L1 or L2
        L2 = 1
        t = np.linspace(0, 30, 750)               # For real time use 750 (same for the
        g = 9.81                                  # frames on 129)

        ## Define state vector ##
        def dSdt(S, t, m1, m2, L1, L2, g):
            theta1, z1, theta2, z2 = S
            return[
                dtheta1dt(z1),
                dz1dt(t, m1, m2, L1, L2, g, theta1, theta2, z1, z2),
                dtheta2dt(z2),
                dz2dt(t, m1, m2, L1, L2, g, theta1, theta2, z1, z2)
            ]

        ## ODE solution w/ state function ##
        ans = odeint(dSdt, y0=[2, -2, 1.5, 2],     # (y0) represents the initial S state vector
                    t=t, args=(m1, m2, L1, L2, g))

        self.theta1_data = ans.T[0]                     # (theta1) and (theta2) fuctions of time
        self.theta2_data = ans.T[2]
        print('asdkjfnkasdfmnsa')

    def shapes(self, mass1, mass2, stick1, stick2):
        self.mass1 = mass1
        self.mass2 = mass2
        self.stick1 = stick1
        self.stick2 = stick2

    def updating_shapes(self, i):
        theta1 = self.theta1[i]
        theta2 = self.theta2[i]

        x1 = L1 * smp.sin(theta1)
        y1 = - L1 * smp.cos(theta1)
        x2 = x1 + L2 * smp.sin(theta2)
        y2 = y1 - L2 * smp.cos(theta2)

        self.mass1.set_center((x1, y1))
        self.mass2.set_center((x2, y2))
        self.stick1.set_data((0,x1), (0, y1))
        self.stick2.set_data((x1,x2), (y1, y2))

pends = DoublePendulum(y0 = [2, -2, 1.5, 2])
pends.pend_ODE_solver(t=t)
print('hello')


