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

# combine data
X = np.concatenate([x_exact, x_approx]).reshape(-1, 1)
y = np.concatenate([y_exact, y_approx])

# ---------------------------
# 3. Define spatial trust weighting
# ---------------------------
def trust_weight(x):
    """
    Full trust in [0, 0.3], linearly decaying to 0.2 by x=1.
    """
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
svr = SVR(kernel='rbf', C=10, gamma=10)
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
