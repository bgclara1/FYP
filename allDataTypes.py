import numpy as np
import matplotlib.pyplot as plt

# True function
def f_true(x):
    return np.sin(2 * np.pi * x) + 0.3 * x

np.random.seed(42)
x_all = np.linspace(0, 1, 200)
y_true = f_true(x_all)

DATA = {}

# 1. Exact
data_exact = (x_all, f_true(x_all))
DATA['exact'] = data_exact

# 2. Y-axis offset
data_y_offset = (x_all, f_true(x_all) + 0.5)
DATA['y_offset'] = data_y_offset

# 3. Y-axis scaled
data_y_scaled = (x_all, 1.5 * f_true(x_all))
DATA['y_scaled'] = data_y_scaled

# 4. X-axis offset
data_x_offset = (x_all, f_true(x_all + 0.1))
DATA['x_offset'] = data_x_offset

# 5. X-axis scaled
data_x_scaled = (x_all, f_true(x_all * 0.8))
DATA['x_scaled'] = data_x_scaled

# 6. Noisy
data_noisy = (x_all, f_true(x_all) + np.random.normal(0, 0.2, len(x_all)))
DATA['noisy'] = data_noisy

# 7. Non-linear distortion 
# y + 0.1sin(y) - exp(y)
data_nl_distort = (x_all, f_true(x_all)+ 0.1 * np.sin(f_true(x_all))-np.exp(f_true(x_all)))
DATA['nonlinear_distortion'] = data_nl_distort

'''
# Plot all data types
plt.figure(figsize=(12, 8))
for i, (key, (x, y)) in enumerate(DATA.items()):
    if key == 'noisy':
        plt.scatter(x, y, label=key, s=10)
    else:
        plt.plot(x, y, label=key)
#plt.plot(x_all, y_true, 'k--', label='True function', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('"Wrong" 1D Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''