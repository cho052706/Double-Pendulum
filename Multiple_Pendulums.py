import numpy as np
import sympy as smp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D
from scipy.integrate import odeint
from matplotlib.colors import hsv_to_rgb

L1 = 2
L2 = 2
N = 5 # Number of simulated pendulums

## Making a pendulum ##
class DoublePendulum():
    def __init__(self, y0):
        self.y0 = y0

    def pend_ODE_solver (self):
        t, m1, m2, L1, L2, g = smp.symbols('t m1 m2 L1 L2 g')

        theta1, theta2 = smp.symbols('theta1, theta2', cls=smp.Function)
        theta1 = theta1(t)
        theta2 = theta2(t)

        x1 = L1 * smp.sin(theta1)
        y1 = - L1 * smp.cos(theta1)
        x2 = x1 + L2 * smp.sin(theta2)
        y2 = y1 - L2 * smp.cos(theta2)

        ## Derivatives ##
        theta1_dot = smp.diff(theta1, t)
        theta2_dot = smp.diff(theta2, t)
        theta1_ddot = smp.diff(theta1_dot, t)
        theta2_ddot = smp.diff(theta2_dot, t)

        x1_dot = smp.diff(x1, t)                 
        x2_dot = smp.diff(x2, t)
        y1_dot = smp.diff(y1, t)
        y2_dot = smp.diff(y2, t)

        ## Lagrangian ##
        T = 1/2 * m1 * (x1_dot**2 + y1_dot**2) + 1/2 * m2 * (x2_dot**2 + y2_dot**2)
        V = m1 * g * y1  + m2 * g * y2
        L = T - V

        LE1 = smp.diff(L, theta1) - smp.diff(smp.diff(L, theta1_dot), t).simplify()
        LE2 = smp.diff(L, theta2) - smp.diff(smp.diff(L, theta2_dot), t).simplify()

        sols = smp.solve([LE1, LE2], [theta1_ddot, theta2_ddot])

        ## Convert to 1st-order ODEs ##
        dz1dt = smp.lambdify((t, m1, m2, L1, L2, g, theta1, theta2, theta1_dot, theta2_dot), 
                            sols[theta1_ddot])
        dtheta1dt = smp.lambdify(theta1_dot, theta1_dot)

        dz2dt = smp.lambdify((t, m1, m2, L1, L2, g, theta1, theta2, theta1_dot, theta2_dot), 
                            sols[theta2_ddot])
        dtheta2dt = smp.lambdify(theta2_dot, theta2_dot)

        ### Setting values ###
        m1 = 1
        m2 = 1
        L1 = 2 # (ax) might be modified for L1 or L2
        L2 = 2
        t = np.linspace(0, 40, 1000) # Must also edit line 145 if changed
        g = 9.81

        self.m1 = m1
        self.m2 = m2

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
        ans = odeint(dSdt, self.y0,
                    t=t, args=(m1, m2, L1, L2, g))

        self.theta1_data = ans.T[0]
        self.theta2_data = ans.T[2]

    def shapes(self, mass1, mass2, stick1, stick2):
        self.mass1 = mass1
        self.mass2 = mass2
        self.stick1 = stick1
        self.stick2 = stick2

    def updating_shapes(self, i):
        theta1 = self.theta1_data[i]
        theta2 = self.theta2_data[i]

        x1 = L1 * smp.sin(theta1)
        y1 = - L1 * smp.cos(theta1)
        x2 = x1 + L2 * smp.sin(theta2)
        y2 = y1 - L2 * smp.cos(theta2)

        self.mass1.set_center((x1, y1))
        self.mass2.set_center((x2, y2))
        self.stick1.set_data((0,x1), (0, y1))
        self.stick2.set_data((x1,x2), (y1, y2))

## Creating pendulums ##
pends = [DoublePendulum(y0=[np.pi/3, -1, np.pi/2+0.01*i/N, 1.5]) for i in range(N)] 
for pendulum in pends:
    pendulum.pend_ODE_solver()

## Setting frame ##
fig = plt.figure()
fig.set_facecolor('k')

ax = fig.add_subplot(aspect = 'equal')
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor('k')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.figure.set_size_inches(9,7)

origin = ax.add_patch(plt.Circle((0, 0), 0.2, color='w', zorder=N+1))
background = ax.add_patch(plt.Rectangle((-15, -15), 30, 30, fc='k', zorder=0))

## Adding pedulums to frame ##
for i,pendulum in enumerate(pends):
    blue = hsv_to_rgb(((150+90*i/N)/360,1,1))

    mass1 = ax.add_patch(plt.Circle((0, 0), 0.2*pendulum.m1, color=blue, zorder=i+1))
    mass2 = ax.add_patch(plt.Circle((0, 0), 0.2*pendulum.m2, color=blue, zorder=i+1))
    stick1 = ax.add_line(Line2D((0, 0), (0, 0), color=blue, lw=1, zorder=i))
    stick2 = ax.add_line(Line2D((0, 0), (0, 0), color=blue, lw=1, zorder=i))
    pendulum.shapes(mass1, mass2, stick1, stick2)

## Animation ##
def animate(i):
    for pendulum in pends:
        pendulum.updating_shapes(i)

plt.rcParams['animation.ffmpeg_path'] = 'C:\\Program Files\\ffmpeg\\ffmpeg-8.0.1-essentials_build\\ffmpeg-8.0.1-essentials_build\\bin\\ffmpeg.exe'

ani = FuncAnimation(fig, animate, frames=1000)
ani.save("many.mp4", writer=FFMpegWriter(fps=25, metadata=dict(artist='Cedric Ho')))

print('done :))')