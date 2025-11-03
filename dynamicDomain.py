""" 
This code has two sets of data. One exact set following the true function on a restricted domain. One with a systematic offset in a separate domain. 
The point of the code is to vary the weighting of the exact data as a function of the domain, and calculate corresponding MSE.
Domain for exact data 0 to 0.3, domain for offset data 0.3 to 1.
Full trust in exact data [0, 0.3], linearly decaying to 0.2 between 0.3 and 1.
200 total data points
Function: y = sin(2*pi*x) + 0.3*x
SVM model RBF kernel. 
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# ---------------------------
# 1. True function
# ---------------------------
def f_true(x):
    return np.sin(2 * np.pi * x) + 0.3 * x

# ---------------------------
# 2. Generate data
# ---------------------------
np.random.seed(0)
x_all = np.linspace(0, 1, 200)
y_true = f_true(x_all)

# exact data (trusted)
x_exact = x_all[x_all <= 0.3]
y_exact = f_true(x_exact)

# approximate data (biased)
x_approx = x_all[x_all > 0.3]
y_approx = f_true(x_approx) + 0.5  # bias
#y_approx_derivs = np.gradient(y_approx, x_approx)

# combine data
X = np.concatenate([x_exact, x_approx]).reshape(-1, 1)
y = np.concatenate([y_exact, y_approx])

# ---------------------------
# 3. Define spatial trust weighting
# ---------------------------
def trust_weight(x):
    x = np.array(x)
    w = np.ones_like(x)
    decay_start, decay_end = 0.3, 1.0
    min_weight = 0.2
    
    mask = x > decay_start
    w[mask] = 1 - (1 - min_weight) * (x[mask] - decay_start) / (decay_end - decay_start)
    w = np.clip(w, min_weight, 1)
    return w

weights = trust_weight(np.concatenate([x_exact, x_approx]))

# ---------------------------
# 4. Train SVR with spatial weights
# ---------------------------
svr = SVR(kernel='rbf')
svr.fit(X, y, sample_weight=weights)

# predict
X_test = x_all.reshape(-1, 1)
y_pred = svr.predict(X_test)

# ---------------------------
# 5. Compute and print MSE
# ---------------------------
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error (MSE): {mse:.6f}")

# ---------------------------
# 6. Plot results
# ---------------------------
plt.figure(figsize=(9,6))
plt.plot(x_all, y_true, 'k-', label='True f(x)')
plt.plot(x_all, y_pred, 'r--', label='Weighted SVR prediction')
plt.scatter(x_exact, y_exact, color='blue', label='Exact data')
plt.scatter(x_approx, y_approx, color='orange', alpha=0.6, label='Approx data')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Spatially weighted SVR (MSE = {mse:.4e})')
plt.legend()
plt.grid(True)
plt.show()
