import numpy as np
import sympy as smp
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.lines import Line2D
from scipy.integrate import odeint

## Variables and functions ##
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
L1 = 1     
L2 = 1
t = np.linspace(0, 30, 750)
g = 9.81                                  

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

theta1_data = ans.T[0]                     # (theta1) and (theta2) fuctions of time
theta2_data = ans.T[2]

## Background setup ##
fig = plt.figure()
fig.set_facecolor('k')

ax = fig.add_subplot(aspect = 'equal')
ax.set_facecolor('k')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-2.25, 2.25)                   # Edit (ax) size here
ax.set_ylim(-2.25, 2.25)

## Circle and line setup ##
origin = ax.add_patch(plt.Circle((0, 0), 0.1, color='w', zorder=3))
mass1 = ax.add_patch(plt.Circle((0, 0), 0.1*m1, color='y', zorder=3))
mass2 = ax.add_patch(plt.Circle((0, 0), 0.1*m2, color='m', zorder=3))
stick1 = ax.add_line(Line2D((0, 0), (0, 0), color='y', lw=1, zorder=2))
stick2 = ax.add_line(Line2D((0, 0), (0, 0), color='m', lw=1, zorder=2))

## Mass trace setup ##
m1_trace = ax.add_line(Line2D((0, 0), (0, 0), color='y', ls='--', lw=0.5, zorder=1))
m2_trace = ax.add_line(Line2D((0, 0), (0, 0), color='m', lw=0.5, zorder=1))
x1_data, y1_data = [], []
x2_data, y2_data = [], []

print('Creating animation...')

## Animation ##
def animate(i):
    theta1 = theta1_data[i]
    theta2 = theta2_data[i]

    x1 = L1 * smp.sin(theta1)
    y1 = - L1 * smp.cos(theta1)
    x2 = x1 + L2 * smp.sin(theta2)
    y2 = y1 - L2 * smp.cos(theta2)

    mass1.set_center((x1, y1))
    mass2.set_center((x2, y2))
    stick1.set_data((0,x1), (0, y1))
    stick2.set_data((x1,x2), (y1, y2))

    x1_data.append(x1)             
    y1_data.append(y1)            
    x2_data.append(x2)                  
    y2_data.append(y2)

    m1_trace.set_data((x1_data, y1_data))
    m2_trace.set_data((x2_data, y2_data))

    return m1_trace, m2_trace

ani = matplotlib.animation.FuncAnimation(fig, animate, frames=750) 
ani.save("double_pen.gif", writer=matplotlib.animation.PillowWriter(fps=25))

print('Done')