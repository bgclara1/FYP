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

# Exact data (trusted)
x_exact = x_all[::10]  # 20 exact points
y_exact = f_true(x_exact)

# Approx data 1 (biased upward)
x_approx1 = x_all
y_approx1 = f_true(x_approx1) + 0.4

# Approx data 2 (biased downward + scaled)
x_approx2 = x_all
y_approx2 = 0.9 * f_true(x_approx2) - 0.2

# Combine for model training
X = np.concatenate([x_exact, x_approx1, x_approx2]).reshape(-1, 1)
y = np.concatenate([y_exact, y_approx1, y_approx2])

# ---------------------------
# 3. Choose some α₁, α₂ combinations to visualise
# ---------------------------
alpha_pairs = [
    (0.0, 0.0),  # trust only exact data
    (0.5, 0.5),  # moderate trust in both
    (1.0, 0.0),  # trust approx1 only
    (0.0, 1.0),  # trust approx2 only
    (1.0, 1.0)   # trust all equally
]

colors = ['purple', 'red', 'orange', 'blue', 'green']
predictions = []
mse_values = []

# ---------------------------
# 4. Train models for each α₁, α₂
# ---------------------------
for (a1, a2) in alpha_pairs:
    w_exact = np.ones_like(y_exact)
    w_approx1 = a1 * np.ones_like(y_approx1)
    w_approx2 = a2 * np.ones_like(y_approx2)
    sample_weights = np.concatenate([w_exact, w_approx1, w_approx2])
    
    svr = SVR(kernel='rbf', C=10, gamma=10)
    svr.fit(X, y, sample_weight=sample_weights)
    
    y_pred = svr.predict(x_all.reshape(-1, 1))
    predictions.append((a1, a2, y_pred))
    
    mse = mean_squared_error(y_true, y_pred)
    mse_values.append(mse)

# ---------------------------
# 5. Plot the real function, datasets, and models
# ---------------------------
plt.figure(figsize=(10, 6))

# True function
plt.plot(x_all, y_true, 'k-', linewidth=2, label='True f(x)')

# Data sets
plt.scatter(x_exact, y_exact, color='black', label='Exact data', zorder=5)
plt.scatter(x_approx1, y_approx1, color='orange', alpha=0.6, label='Approx data 1 (+0.4 bias)', s=20)
plt.scatter(x_approx2, y_approx2, color='cyan', alpha=0.6, label='Approx data 2 (scaled -0.2)', s=20)

# Model predictions
for (color, (a1, a2, y_pred)) in zip(colors, predictions):
    plt.plot(x_all, y_pred, '--', color=color, linewidth=1.8,
             label=f'Model α₁={a1:.1f}, α₂={a2:.1f}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('True function, data sources, and SVR predictions for different α₁, α₂')
plt.legend(ncol=2)
plt.grid(True)
plt.show()

# ---------------------------
# 6. Print MSE values
# ---------------------------
for (a1, a2), mse in zip(alpha_pairs, mse_values):
    print(f"α₁={a1:.1f}, α₂={a2:.1f}  →  MSE = {mse:.5f}")
