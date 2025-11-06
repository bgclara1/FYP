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
from allDataTypes import DATA

# 1. Load data
np.random.seed(0)
x_all, y_true = DATA['exact']
x_exact, y_exact = DATA['exact']
x_approx, y_approx = DATA['y_offset']

# 2. Prepare domain masks
mask_exact = x_exact <= 0.3
x_exact = x_exact[mask_exact]
y_exact = y_exact[mask_exact]
mask_approx = x_approx > 0.3
x_approx = x_approx[mask_approx]
y_approx = y_approx[mask_approx]
y_approx_derivs = np.gradient(y_approx)

# 3. Combine for model training
X = np.concatenate([x_exact, x_approx]).reshape(-1, 1)
y = np.concatenate([y_exact, y_approx_derivs])

# 4. Train SVR
svr = SVR(kernel='rbf')
svr.fit(X, y)
X_test = x_all.reshape(-1, 1)
y_pred = svr.predict(X_test)

# 5. Compute and print MSE
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error (MSE): {mse:.6f}")

# 6. Plot results
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

