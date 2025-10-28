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
y_approx = f_true(x_approx) + 0.5   # systematic offset

# combine
X = np.concatenate([x_exact, x_approx]).reshape(-1, 1)
y = np.concatenate([y_exact, y_approx])

# test domain (true function)
X_test = x_all.reshape(-1, 1)
y_test = y_true

# ---------------------------
# 3. Loop over alphas
# ---------------------------
alphas = np.linspace(0, 1, 10)  # 0, 0.2, 0.4, 0.6, 0.8, 1
errors = []
predictions = []

for alpha in alphas:
    # assign weights
    w_exact = np.ones_like(y_exact)
    w_approx = alpha * np.ones_like(y_approx)
    sample_weights = np.concatenate([w_exact, w_approx])
    
    # train SVR
    svr = SVR(kernel='rbf', C=10, gamma=10)
    svr.fit(X, y, sample_weight=sample_weights)
    
    # predict
    y_pred = svr.predict(X_test)
    predictions.append(y_pred)
    
    # error vs true function
    mse = mean_squared_error(y_test, y_pred)
    errors.append(mse)

# ---------------------------
# 4. Plot all predictions together
# ---------------------------
plt.figure(figsize=(9,6))

# true function
plt.plot(x_all, y_true, 'k-', linewidth=2, label='True f(x)')

# each SVR prediction
for alpha, y_pred in zip(alphas, predictions):
    plt.plot(x_all, y_pred, '--', linewidth=1.5, label=f'α={alpha:.1f}')

# data points
plt.scatter(x_exact, y_exact, color='blue', label='Exact data')
plt.scatter(x_approx, y_approx, color='orange', label='Approx data', alpha=0.6)

plt.xlabel('x')
plt.ylabel('y')
plt.title('SVR fits for varying trust in approximate data (α)')
plt.legend(ncol=2)
plt.grid(True)
plt.show()

# ---------------------------
# 5. Also show MSE vs α
# ---------------------------
plt.figure(figsize=(7,5))
plt.plot(alphas, errors, 'o-', lw=2)
plt.xlabel('Trust in approximate data (α)')
plt.ylabel('Mean Squared Error vs true f(x)')
plt.title('Effect of α on accuracy')
plt.grid(True)
plt.show()

best_alpha = alphas[np.argmin(errors)]
print(f"Best α = {best_alpha:.2f}, MSE = {min(errors):.4e}")
