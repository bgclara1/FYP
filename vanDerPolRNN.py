import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from vdpData import vdp_datasets

# pick your trajectory
(t1, x1, v1) = vdp_datasets[(3.0, 1.0)]  # mu=3, a=1

t1 = np.asarray(t1)
x1 = np.asarray(x1)

# ---- define exact split by time ----
t_train_end = 24.0
t_val_end   = 32.0
t_pred_end  = 40.0  # just for clarity

idx_train_end = np.searchsorted(t1, t_train_end)  # first index with t >= 24
idx_val_end   = np.searchsorted(t1, t_val_end)    # first index with t >= 32
idx_pred_end  = np.searchsorted(t1, t_pred_end)   # first index with t >= 40 (should be len(t1))

x_train_raw = x1[:idx_train_end]           # [0, 24)
x_valid_raw = x1[idx_train_end:idx_val_end]  # [24, 32)

# normalise using *train* stats only
x_mean = x_train_raw.mean()
x_std  = x_train_raw.std()

x_train = (x_train_raw - x_mean) / x_std
x_valid = (x_valid_raw - x_mean) / x_std

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.cast(series, dtype=tf.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices(series)
    
    dataset = dataset.window(window_size + 1,
                             shift=1,
                             drop_remainder=True)
    
    dataset = dataset.flat_map(lambda w: w.batch(window_size + 1))
    
    dataset = dataset.shuffle(buffer_size=max(1, shuffle_buffer))
    
    # (window[:-1] as input sequence, window[-1] as target)
    dataset = dataset.map(lambda w: (w[:-1], w[-1]))
    
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

window_size = 50
batch_size = 32
shuffle_buffer_size = 1000

train_dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
test_dataset  = windowed_dataset(x_valid, window_size, batch_size, 1)

def build_model():
    optimizer = tf.keras.optimizers.Adam()
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(
            lambda x: tf.expand_dims(x, axis=-1),  # (batch, time) -> (batch, time, 1)
            input_shape=[None]
        ),
        tf.keras.layers.SimpleRNN(units=64),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']   # MAPE is nasty near zero
    )
    return model

model = build_model()
model.summary()

def train_model(train_dataset):
    start_time = time.time()
    
    history = model.fit(
        train_dataset,
        epochs=35,
        validation_data=test_dataset
    )
    
    end_time = time.time()
    runtime = end_time - start_time
    return runtime, history, model

runtime, history, model = train_model(train_dataset)
print("Training time:", runtime, "seconds")

# full normalised series up to 32 (train + val)
x_up_to_32_raw = x1[:idx_val_end]
x_up_to_32 = (x_up_to_32_raw - x_mean) / x_std

# how many steps correspond to 8 time units?
dt = t1[1] - t1[0]
n_pred_steps = idx_pred_end - idx_val_end  # number of indices between 32 and 40

# initialise window: last window_size points ending at t=32-
full_series = x_up_to_32
window = full_series[-window_size:].tolist()

pred_norm = []

for _ in range(n_pred_steps):
    inp = np.array(window, dtype=np.float32).reshape(1, -1)  # shape (1, window_size)
    next_norm = model.predict(inp, verbose=0)[0, 0]
    pred_norm.append(next_norm)
    window.append(next_norm)
    window.pop(0)

# denormalise predictions
pred = np.array(pred_norm) * x_std + x_mean

# true data over [32, 40]
t_true_32_40 = t1[idx_val_end:idx_pred_end]
x_true_32_40 = x1[idx_val_end:idx_pred_end]

# time grid for predictions (matches those indices)
t_pred_32_40 = t_true_32_40  # same spacing / indices

# ---- plot ----
plt.figure(figsize=(9, 5))

# optionally: full true curve faintly
plt.plot(t1, x1, alpha=0.3, label="True x(t) (0–40)")

# focus: segment 32–40
plt.plot(t_true_32_40, x_true_32_40, label="True x(t), 32–40")
plt.plot(t_pred_32_40, pred, '--', label="RNN prediction, 32–40")

plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("RNN forecast vs true Van der Pol: last 8 time units")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
