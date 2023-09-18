from typing import List, Tuple
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


np.random.seed(1337)


def get_data(nsamples: int = 100) -> Tuple[np.array, np.array]:
    x = np.linspace(0, 10, nsamples)
    y = 2 * x + 3.5
    return (x, y)


def add_noise(y: np.array, std: int) -> np.array:
    noise = np.random.normal(scale=std, size=y.size)
    return y + noise


def gradient_descent(X: np.array, y: np.array, 
                     lr: float = 0.01, 
                     epoch: int = 1000) -> Tuple[
                                            float, 
                                            float, 
                                            List[Tuple[float, float]],
                                            List[float]]:
    """
    Gradient Descent for a single feature
    """

    m, b = 0.33, 0.48  # initial guess for parameters
    log, mse = [], []  # lists to store learning process
    N = len(X)         # number of samples
    
    for _ in range(epoch):
                
        f = y - (m*X + b)
    
        # Updating m and b
        m -= lr * (-2 * X.dot(f).sum() / N)
        b -= lr * (-2 * f.sum() / N)
        
        log.append((m, b))
        mse.append(mean_squared_error(y, (m*X + b)))        
    
    return m, b, log, mse


def run_experiment(outliers: int, noise_std: int, ns: List[int], lrs: List[int]):
    # Getting data
    x, y_true = get_data()
    y = add_noise(y_true, std=noise_std)
    if outliers:
        outliers_indexes = np.random.choice([i for i in range(len(y))], size=outliers)
        y[outliers_indexes] = add_noise(y[outliers_indexes], std=50)
        

    # Plot and investigate data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o", label="data")
    ax.legend(loc="best")
    plt.savefig("data.png")

    fig1, ax1 = plt.subplots(nrows=len(ns), ncols=len(lrs), figsize=(20, 15))
    fig1.suptitle(f"Noise std = {noise_std}, {outliers} outliers")

    fig2, ax2 = plt.subplots(nrows=len(ns), ncols=len(lrs), figsize=(20, 15))
    fig2.suptitle(f"Noise std = {noise_std}, {outliers} outliers")
    for i, epoches in enumerate(ns):
        for j, lr in enumerate(lrs):
            m, b, log, mse = gradient_descent(x, y, lr=lr, epoch=epoches)
            print(f"Params (with noise = {noise_std}, n = {epoches}, learning rate = {lr}):\n\t{m = }\n\t{b = }")
            
            ax1[i, j].set_title(f"Lr = {lr}, epoches = {epoches}")
            ax1[i, j].plot(x, y, "o", label="data")
            ax1[i, j].plot(x, y_true, "b-", label="True")
            ax1[i, j].plot(x, m*x+b, "r--.", label="Gradient Descent")
            ax1[i, j].legend(loc="best")

            ax2[i, j].set_title(f"Lr = {lr}, epoches = {epoches}")
            ax2[i, j].plot([x for x in range(1, epoches+1)], mse)
            ax2[i, j].set_xlabel('Epoch')
            ax2[i, j].set_ylabel('MSE')
    fig1.savefig(f"gd_regression_noise.png")
    fig2.savefig(f"mse_vs_epoches.png")


if __name__ == "__main__":
    run_experiment(outliers=1, noise_std=1, ns=[10, 200, 10000], lrs=[0.01, 0.001, 0.0001])