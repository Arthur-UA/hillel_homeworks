from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


np.random.seed(1337)
errors = []


def get_linear_data(nsamples: int = 50) -> Tuple[np.array, np.array]:
    x = np.linspace(0, 10, nsamples)
    y = 2 * x + 3.5
    return (x, y)


def get_non_linear_data(nsamples: int = 150) -> Tuple[np.array, np.array]:
    x = np.linspace(0, 10, nsamples)
    y = 4 * x**2 + 3 * x + 8
    return (x, y)


def add_noise(y: np.array) -> np.array:
    noise = np.random.normal(scale=1.5, size=y.size)
    return y + noise


def mae_regression(guess: np.array, x: np.array, y: np.array, linear: bool = True) -> float:
    """MAE Minimization Regression"""
    if linear:
        m = guess[0]
        b = guess[1]
        y_hat = m * x + b
    else:
        m1 = guess[0]
        m2 = guess[1]
        b = guess[2]
        y_hat = m1 * x**2 + m2 * x + b
    
    # Get loss MSE
    mae = (np.abs(y - y_hat)).mean()
    errors.append(mae)
    return mae


def mse_regression(guess: np.array, x: np.array, y: np.array, linear: bool = True) -> float:
    """MSE Minimization Regression"""
    if linear:
        m = guess[0]
        b = guess[1]
        y_hat = m * x + b
    else:
        m1 = guess[0]
        m2 = guess[1]
        b = guess[2]
        y_hat = m1 * x**2 + m2 * x + b

    # Get loss MSE
    mse = (np.square(y - y_hat)).mean()
    errors.append(mse)
    return mse


def rmse_regression(guess: np.array, x: np.array, y: np.array, linear: bool = True) -> float:
    """RMSE Minimization Regression"""
    if linear:
        m = guess[0]
        b = guess[1]
        y_hat = m * x + b
    else:
        m1 = guess[0]
        m2 = guess[1]
        b = guess[2]
        y_hat = m1 * x**2 + m2 * x + b

    # Get loss MSE
    rmse = np.sqrt((np.square(y - y_hat)).mean())
    errors.append(rmse)
    return rmse


def mape_regression(guess: np.array, x: np.array, y: np.array, linear: bool) -> float:
    """Mean absolute persentage error"""
    if linear:
        m = guess[0]
        b = guess[1]
        y_hat = m * x + b
    else:
        m1 = guess[0]
        m2 = guess[1]
        b = guess[2]
        y_hat = m1 * x**2 + m2 * x + b

    # Get loss MSE
    mape = (np.abs((y - y_hat) / y)).mean()
    errors.append(mape)
    return mape


def experiment(n_samples: int, methods: list, linear: bool):
    # Getting data
    x, y_true = get_linear_data(n_samples) if linear else get_non_linear_data(n_samples)
    y = add_noise(y_true)

    # Plot and investigate data

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o", label="data")
    ax.legend(loc="best")
    plt.savefig("data.png")

    initial_guess = np.array([5, -3]) if linear else np.array([5, -3, 12])
    
    # Maximizing the probability for point to be from the distribution
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,6))
    fig.suptitle(f'Observations = {n_samples}')
    for i, method in enumerate(methods):
        results = minimize(
            mape_regression,
            initial_guess,
            args=(x, y, linear,),
            method=method,
            options={"disp": True})
        
        ax[i].set_title(f'{method}')
        ax[i].plot([x for x in range(len(errors))], errors)
        ax[i].set_xlabel('n-iterations')
        ax[i].set_ylabel('Error')
        plt.savefig("error_graph.png")
        print("Parameters: ", results.x)
        errors.clear()

    xx = np.linspace(np.min(x), np.max(x), 10)
    if linear:
        yy = results.x[0] * xx + results.x[1] 
    else:
        yy = results.x[0] * xx**2 + results.x[1] * xx + results.x[2]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o", label="data")
    ax.plot(x, y_true, "b-", label="True")
    ax.plot(xx, yy, "r--.", label="MLE")
    ax.legend(loc="best")
    plt.savefig("mse_regression.png")


if __name__ == "__main__":
    experiment(10000, ['Nelder-Mead', 'CG', 'bfgs'], linear=False)