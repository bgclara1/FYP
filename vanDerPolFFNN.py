import matplotlib.pyplot as plt
import numpy as np
import time
from tensorflow import keras
from vdpData import vdp_datasets

plt.figure(figsize=(9, 5))

# ---- pull the two datasets you want ----
(t1, x1, v1) = vdp_datasets[(3.0, 1.0)]
(t2, x2, v2) = vdp_datasets[(3.5, 1.5)]

# ---- plot ----
plt.plot(t1, x1, label="mu = 3.0, a = 1.0")
plt.plot(t2, x2, label="mu = 3.5, a = 1.5")

plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Van der Pol: selected cases")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -------------

# Input: time t
t_all = t1.astype("float32").reshape(-1, 1)   # shape (N, 1)

# Output: [x(t), v(t)]
y_all = np.stack([x1, v1], axis=1).astype("float32")  # shape (N, 2)

# Simple train/val split
split_idx = int(0.8 * len(t_all))
train_dataset = t_all[:split_idx]
train_labels  = y_all[:split_idx]
val_dataset   = t_all[split_idx:]
val_labels    = y_all[split_idx:]

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(1,)),  # t only
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(2)  # outputs: x, v
    ])
    return model

model = build_model()
model.summary()

def train_model(model, train_dataset, train_labels, val_dataset, val_labels):
    start_time = time.time()
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
        loss='mse',
        metrics=['mse', 'mape']
    )
    
    history = model.fit(
        train_dataset,
        train_labels,
        epochs=700,
        validation_data=(val_dataset, val_labels),
        verbose=1  # change to 1 if you want to see the progress
    )
    
    end_time = time.time()
    runtime = end_time - start_time
    return runtime, history, model

runtime, history, model = train_model(
    model, train_dataset, train_labels, val_dataset, val_labels
)
print("Training time:", runtime, "seconds")

# Create new time grid: last t → last t + 20
t_start_extra = t1[-1]
t_end_extra   = t1[-1] + 20.0
t_extra = np.linspace(t_start_extra, t_end_extra, 400).astype("float32").reshape(-1, 1)

y_extra = model.predict(t_extra)
x_extra = y_extra[:, 0]
v_extra = y_extra[:, 1]

plt.figure(figsize=(9, 5))
plt.plot(t1, x1, label="True x(t) (train domain)")
plt.plot(t_extra.squeeze(), x_extra, label="NN x(t) (extra 20t)")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
