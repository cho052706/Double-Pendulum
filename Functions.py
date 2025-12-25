import ODE_Solving

def dSdt(S, m1, m2, L1, L2, t, g):
    theta1, z1, theta2, z2 = S
    return[
        ODE_Solving.dz1dt(m1, m2, L1, L2, t, g, theta1, theta2, z1, z2),
        ODE_Solving.dtheta1dt(z1),
        ODE_Solving.dz2dt(m1, m2, L1, L2, t, g, theta1, theta2, z1, z2),
        ODE_Solving.dtheta2dt(z2)
    ]