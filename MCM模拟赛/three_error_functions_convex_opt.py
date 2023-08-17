import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rc('font', family='Times New Roman', size=15)
plt.style.use('seaborn')  # 是seaborn的暗色系
sns.set_theme(style="darkgrid")  # 改为黑暗模式


# MSE loss function
def mse_loss(y_pred, y_true):
    squared_error = (y_pred - y_true) ** 2
    sum_squared_error = np.sum(squared_error)
    loss = sum_squared_error / y_true.size
    return loss


# Plotting
x_vals = np.arange(-3, 3, 0.01)
y_vals = np.square(x_vals)

plt.plot(x_vals, y_vals, label='MSE loss',c="b")
plt.legend()
plt.grid(True, which="major")


# MAE loss function
def mae_loss(y_pred, y_true):
    abs_error = np.abs(y_pred - y_true)
    sum_abs_error = np.sum(abs_error)
    loss = sum_abs_error / y_true.size
    return loss


# Plotting
x_vals = np.arange(-3, 3, 0.01)
y_vals = np.abs(x_vals)

plt.plot(x_vals, y_vals,label='MAE loss',c = "red")
plt.legend()
plt.grid(True, which="major")


# Huber loss function
def huber_loss(y_pred, y, delta=1.0):
    huber_mse = 0.5 * (y - y_pred) ** 2
    huber_mae = delta * (np.abs(y - y_pred) - 0.5 * delta)
    return np.where(np.abs(y - y_pred) <= delta, huber_mse, huber_mae)


# Plotting
x_vals = np.arange(-3, 3, 0.01)

delta = 1.5
huber_mse = 0.5 * np.square(x_vals)
huber_mae = delta * (np.abs(x_vals) - 0.5 * delta)
y_vals = np.where(np.abs(x_vals) <= delta, huber_mse, huber_mae)

plt.plot(x_vals, y_vals,label = 'Huber loss',c = "green")
plt.legend()
plt.grid(True, which="major")
plt.show()
