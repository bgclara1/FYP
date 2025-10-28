import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

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

# exact data only in first third of domain
mask_exact = x_all <= 1/3
x_exact = x_all[mask_exact][::4]
y_exact = f_true(x_exact)

# approximate data
x_approx1 = x_all
y_approx1 = f_true(x_approx1) + 0.5             # scaled
x_approx2 = x_all
y_approx2 = f_true(x_approx2) + np.random.normal(0, 0.25, len(x_approx2))  # noisy

# combine
X = np.concatenate([x_exact, x_approx1, x_approx2]).reshape(-1, 1)
y = np.concatenate([y_exact, y_approx1, y_approx2])

# ---------------------------
# 3. Define α₁, α₂ ranges
# ---------------------------
alphas1 = np.linspace(0, 1, 20)   # trust in scaled data
alphas2 = np.linspace(0, 1, 20)   # trust in noisy data
errors = np.zeros((len(alphas1), len(alphas2)))

# ---------------------------
# 4. Double loop: compute MSE for each (α₁, α₂)
# ---------------------------
for i, a1 in enumerate(alphas1):
    for j, a2 in enumerate(alphas2):
        w_exact = np.ones_like(y_exact)
        w_approx1 = a1 * np.ones_like(y_approx1)
        w_approx2 = a2 * np.ones_like(y_approx2)
        weights = np.concatenate([w_exact, w_approx1, w_approx2])

        svr = SVR(kernel='rbf', C=10, gamma=10)
        svr.fit(X, y, sample_weight=weights)
        y_pred = svr.predict(x_all.reshape(-1, 1))
        mse = mean_squared_error(y_true, y_pred)
        errors[i, j] = mse

# ---------------------------
# 5. 3D surface plot
# ---------------------------
A1, A2 = np.meshgrid(alphas1, alphas2)

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(A1, A2, errors.T, cmap='viridis', edgecolor='none', alpha=0.9)
ax.set_xlabel('Trust in scaled data (α₁)')
ax.set_ylabel('Trust in noisy data (α₂)')
ax.set_zlabel('Mean Squared Error')
ax.set_title('3D MSE Surface vs α₁ and α₂')
fig.colorbar(surf, ax=ax, shrink=0.6, label='MSE')
plt.show()

# ---------------------------
# 6. Print best combination
# ---------------------------
min_idx = np.unravel_index(np.argmin(errors), errors.shape)
best_a1 = alphas1[min_idx[0]]
best_a2 = alphas2[min_idx[1]]
best_mse = errors[min_idx]

print("="*60)
print(f"Lowest MSE = {best_mse:.6f}")
print(f"Best α₁ (scaled) = {best_a1:.2f}, Best α₂ (noisy) = {best_a2:.2f}")
print("="*60)

# ---------------------------
# 7. Graph of all data
# ---------------------------

plt.figure(figsize=(8,5))
plt.plot(x_all, y_true, 'k-', lw=2, label='True f(x)')
plt.scatter(x_exact, y_exact, color='black', s=30, label='Exact data (first third)')
plt.scatter(x_approx1, y_approx1, color='orange', alpha=0.6, s=25, label='Approx 1 (scaled +0.2)')
plt.scatter(x_approx2, y_approx2, color='cyan', alpha=0.6, s=25, label='Approx 2 (noisy)')
plt.legend(); plt.xlabel('x'); plt.ylabel('y'); plt.title('True and approximate datasets'); plt.grid(True); plt.show()
