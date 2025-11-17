# vdp_data.py

import numpy as np
from scipy.integrate import solve_ivp

# Generalized Van der Pol:
#   x'' - μ(1 - a x^2)x' + b x = 0
# We'll fix b = 1.0 and vary mu and a.

def vdp_general(t, y, mu, a, b=1.0):
    x, v = y
    dxdt = v
    dvdt = mu * (1.0 - a * x**2) * v - b * x
    return [dxdt, dvdt]

# Time grid and initial condition
t_start = 0.0
t_end   = 40.0
points  = 2000
t_eval  = np.linspace(t_start, t_end, points)

y0 = [1.0, 0.0]  # x(0), x'(0)

# Parameters:
mu_values = [2.5, 3.0, 3.5]
a_values  = [0.5, 1.0, 1.5]
b_value   = 1.0

# Dictionary to store all datasets:
# vdp_datasets[(mu, a)] = (t, x, v)
vdp_datasets = {}

for mu in mu_values:
    for a in a_values:
        sol = solve_ivp(
            lambda t, y: vdp_general(t, y, mu, a, b_value),
            (t_start, t_end),
            y0,
            t_eval=t_eval
        )

        t = sol.t
        x = sol.y[0]
        v = sol.y[1]

        vdp_datasets[(mu, a)] = (t, x, v)
