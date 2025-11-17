import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from allDataTypes import DATA

# --- assume you already have: x_all, y_true, xv, yv, xg, y_approx, gy ---

x_all, y_true = DATA['exact']
xv = x_all[x_all <= 0.3]
yv = y_true[x_all <= 0.3]
xg = x_all[x_all > 0.3]
y_approx = DATA['noisy'][1][x_all > 0.3]
dx = np.diff(xg)
gy = np.empty_like(xg)
gy[1:-1] = (y_approx[2:] - y_approx[:-2]) / (xg[2:] - xg[:-2])
gy[0] = (y_approx[1] - y_approx[0]) / (xg[1] - xg[0])
gy[-1] = (y_approx[-1] - y_approx[-2]) / (xg[-1] - xg[-2])


# Hyperparameters
value_weight   = 1.0     # how much to trust value data yv
grad_weight    = 0.1     # how much to trust gradient labels gy
lambda_w       = 0.0     # ridge on w (0 = ordinary least squares)

# Design for values:  f(x) = w0 + w1*x
A_v = np.column_stack([np.ones_like(xv), xv])     # shape (nv, 2)
b_v = yv

# Design for gradients: f'(x) = w1  (independent of x)
A_g = np.column_stack([np.zeros_like(xg), np.ones_like(xg)])  # shape (ng, 2)
b_g = gy

# Stack with sqrt-weights (equivalent to weighted least squares)
Av = np.sqrt(value_weight) * A_v
bg = np.sqrt(grad_weight)  * b_g
Ag = np.sqrt(grad_weight)  * A_g
bv = np.sqrt(value_weight) * b_v

A = np.vstack([Av, Ag])           # (nv+ng, 2)
b = np.concatenate([bv, bg])      # (nv+ng,)

# Ridge-regularized normal equations  (set lambda_w=0 for plain OLS)
AtA = A.T @ A + lambda_w * np.eye(2)
Atb = A.T @ b
w0, w1 = solve(AtA, Atb)          # coefficients

def predict_linear(x):
    x = np.atleast_1d(x)
    return w0 + w1 * x

y_pred = predict_linear(x_all)

# --- Plot
plt.figure(figsize=(8,5))
plt.plot(x_all, y_true, 'k-', label='True f(x)')
plt.plot(x_all, y_pred, 'r--', label='Linear fit (no kernel)')
plt.scatter(xv, yv, color='blue', s=25, label='Exact data (values)')
plt.scatter(xg, y_approx, color='orange', s=25, label='Approx data (shown only)')
plt.xlabel('x'); plt.ylabel('y'); plt.title('Linear regression with value + gradient constraints')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

print(f"w0 = {w0:.4f}, w1 = {w1:.4f}")
