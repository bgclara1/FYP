"""
This code has two sets of data. One exact set following the true function on a restricted domain. One with a systematic offset in a separate domain.
The point of the code is to vary the weighting of the offset data and track the MSE with each.
Domain for exact data 0 to 0.3, domain for offset data 0.3 to 1
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

# 3. Combine for model training
X = np.concatenate([x_exact, x_approx]).reshape(-1, 1)
y = np.concatenate([y_exact, y_approx])
X_test = x_all.reshape(-1, 1)
y_test = y_true

# 4. Loop over alphas
alphas = np.linspace(0, 1, 10)
errors = []
predictions = []
for alpha in alphas:
    w_exact = np.ones_like(y_exact)
    w_approx = alpha * np.ones_like(y_approx)
    sample_weights = np.concatenate([w_exact, w_approx])
    svr = SVR(kernel='rbf')
    svr.fit(X, y, sample_weight=sample_weights)
    y_pred = svr.predict(X_test)
    predictions.append(y_pred)
    mse = mean_squared_error(y_test, y_pred)
    errors.append(mse)

# 5. Plot all predictions together
plt.figure(figsize=(9,6))
plt.plot(x_all, y_true, 'k-', linewidth=2, label='True f(x)')
for alpha, y_pred in zip(alphas, predictions):
    plt.plot(x_all, y_pred, '--', linewidth=1.5, label=f'α={alpha:.1f}')
plt.scatter(x_exact, y_exact, color='blue', label='Exact data')
plt.scatter(x_approx, y_approx, color='orange', label='Approx data', alpha=0.6)
plt.xlabel('x')
plt.ylabel('y')
plt.title('SVR fits for varying trust in approximate data (α)')
plt.legend(ncol=2)
plt.grid(True)
plt.show()

# 6. Show MSE vs α
plt.figure(figsize=(7,5))
plt.plot(alphas, errors, 'o-', lw=2)
plt.xlabel('Trust in approximate data (α)')
plt.ylabel('Mean Squared Error vs true f(x)')
plt.title('Effect of α on accuracy')
plt.grid(True)
plt.show()

best_alpha = alphas[np.argmin(errors)]
print(f"Best α = {best_alpha:.2f}, MSE = {min(errors)::.4e}")

# 7. Report best result
best_idx = np.argmin(errors)
best_alpha = alphas[best_idx]
best_mse = errors[best_idx]
print("="*50)
print(f"Lowest Mean Squared Error: {best_mse:.6f}")
print(f"Achieved at α = {best_alpha:.2f}")
print("="*50)
