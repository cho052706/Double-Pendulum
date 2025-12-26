import ODE_Solving as ode
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.lines import Line2D

fig = plt.figure()
ax = fig.add_subplot(aspect = 'equal')
ax.set_xlim(-2.25, 2.25)
ax.set_ylim(-2.25, 2.25)
ax.set_xticks([])
ax.set_yticks([])

x1 = 1
y1 = -1
x2 = 2
y2 = -1

origin = ax.add_patch(plt.Circle((0, 0), 0.1, color='black'))
mass1 = ax.add_patch(plt.Circle((x1, y1), 0.1, color='black'))
mass2 = ax.add_patch(plt.Circle((x2, y2), 0.1, color='black'))
stick1 = ax.add_line(Line2D((0, x1), (0, y1), color='black', linewidth=2))
stick2 = ax.add_line(Line2D((x1, x2), (y1, y2), color='black', linewidth=2))


plt.show()