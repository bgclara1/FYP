import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
from allDataTypes import DATA

# 1. RBF kernel and derivatives
def K(x, y, sigma):
    x = np.atleast_1d(x)[:, None]
    y = np.atleast_1d(y)[None, :]
    r2 = (x - y)**2
    return np.exp(-0.5 * r2 / sigma**2)

def dK_dx(x, y, sigma):
    x = np.atleast_1d(x)[:, None]
    y = np.atleast_1d(y)[None, :]
    base = K(x.ravel(), y.ravel(), sigma)
    return - (x - y) / sigma**2 * base

def dK_dy(x, y, sigma):
    x = np.atleast_1d(x)[:, None]
    y = np.atleast_1d(y)[None, :]
    base = K(x.ravel(), y.ravel(), sigma)
    return + (x - y) / sigma**2 * base

def d2K_dxdy(x, y, sigma):
    x = np.atleast_1d(x)[:, None]
    y = np.atleast_1d(y)[None, :]
    r = (x - y)
    base = K(x.ravel(), y.ravel(), sigma)
    return ((1.0 / sigma**2) - (r**2 / sigma**4)) * base

# 2. Load data
x_all, y_true = DATA['exact']
xv = x_all[x_all <= 0.3]
yv = y_true[x_all <= 0.3]
xg = x_all[x_all > 0.3]
y_approx = DATA['noisy'][1][x_all > 0.3]

# 3. Numerical gradient of approx data (central differences)
dx = np.diff(xg)
gy = np.empty_like(xg)
gy[1:-1] = (y_approx[2:] - y_approx[:-2]) / (xg[2:] - xg[:-2])
gy[0] = (y_approx[1] - y_approx[0]) / (xg[1] - xg[0])
gy[-1] = (y_approx[-1] - y_approx[-2]) / (xg[-1] - xg[-2])

sigma = 0.01
lam_v = 0
lam_g = 0
nv, ng = len(xv), len(xg)

# 4. Build blocks with correct shapes
Kvv = K(xv, xv, sigma)
Kvg = dK_dy(xv, xg, sigma)
Kgv = dK_dx(xg, xv, sigma)
Kgg = d2K_dxdy(xg, xg, sigma)

# 5. Assemble big system
A = np.block([
    [Kvv + lam_v*np.eye(nv), Kvg],
    [Kgv,                    Kgg + lam_g*np.eye(ng)]
])
b = np.concatenate([yv, gy])

alpha = solve(A, b)

def predict(x_new):
    x_new = np.atleast_1d(x_new)
    Kv = K(x_new, xv, sigma)
    Kg = dK_dy(x_new, xg, sigma)
    return np.hstack([Kv, Kg]) @ alpha

y_pred = predict(x_all)

# 6. Plot results
plt.figure(figsize=(8, 5))
plt.plot(x_all, y_true, 'k-', label='True f(x)')
plt.plot(x_all, y_pred, 'r--', label='KRR fit')
plt.scatter(xv, yv, color='blue', s=25, label='Exact data')
plt.scatter(xg, y_approx, color='orange', s=25, label='Approx data (offset)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Noisy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
