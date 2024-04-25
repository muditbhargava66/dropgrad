"""
Visualizing the Behavior of DropGrad with Various Optimizers on Optimization Benchmarks
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import rosen
from dropgrad import DropGrad
import torch
from torch.optim import SGD, Adam, AdamW, Adagrad, Adadelta

def rastrigin(x):
    A = 10
    n = len(x)
    return A * n + sum(x**2 - A * np.cos(2 * np.pi * x))

def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    cos_term = -np.exp(np.sum(np.cos(c * x)) / d)
    return a + np.exp(1) + sum_sq_term + cos_term

def apply_dropgrad(optimizer, drop_rate):
    """
    Apply DropGrad to the optimizer if drop_rate is greater than 0.
    """
    if drop_rate > 0:
        return DropGrad(optimizer, drop_rate=drop_rate)
    return optimizer

def optimize(optimizer, x, benchmark_func, num_iterations):
    """
    Run the optimizer on the benchmark function for a given number of iterations.
    """
    trajectory = [x.detach().numpy().copy()]

    for _ in range(num_iterations):
        optimizer.zero_grad()
        y = benchmark_func(x.detach().numpy())
        y_tensor = torch.tensor(y, requires_grad=True)
        y_tensor.backward()
        optimizer.step()
        trajectory.append(x.detach().numpy().copy())

    return trajectory

def visualize_benchmark(benchmark_func, optimizers, num_iterations, drop_rates):
    """
    Visualize the optimization trajectories for different optimizers and drop rates.
    """
    num_optimizers = len(optimizers)
    num_drop_rates = len(drop_rates)

    fig, axs = plt.subplots(num_optimizers, num_drop_rates, figsize=(12, 8), sharex=True, sharey=True)

    if num_optimizers == 1 and num_drop_rates == 1:
        axs = [[axs]]
    elif num_optimizers == 1:
        axs = [axs]
    elif num_drop_rates == 1:
        axs = [[ax] for ax in axs]

    for i, (optimizer_name, base_optimizer) in enumerate(optimizers.items()):
        for j, drop_rate in enumerate(drop_rates):
            x = torch.randn(2, requires_grad=True)
            optimizer = apply_dropgrad(base_optimizer, drop_rate)
            trajectory = optimize(optimizer, x, benchmark_func, num_iterations)

            x_values = [point[0] for point in trajectory]
            y_values = [point[1] for point in trajectory]

            axs[i][j].plot(x_values, y_values, marker='o', markersize=2, linestyle='-', linewidth=0.5)
            axs[i][j].set_title(f"{optimizer_name} (Drop Rate: {drop_rate})")

    fig.suptitle(f"Optimization Trajectories on {benchmark_func.__name__}", fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    num_iterations = 1000
    optimizers = {
        "SGD": SGD([torch.randn(2, requires_grad=True)], lr=0.01),
        "Adam": Adam([torch.randn(2, requires_grad=True)], lr=0.01),
        "AdamW": AdamW([torch.randn(2, requires_grad=True)], lr=0.01),
        "Adagrad": Adagrad([torch.randn(2, requires_grad=True)], lr=0.01),
        "Adadelta": Adadelta([torch.randn(2, requires_grad=True)], lr=0.01),
    }
    drop_rates = [0.0, 0.1, 0.2]
    benchmarks = [rosen, rastrigin, ackley]

    for benchmark_func in benchmarks:
        visualize_benchmark(benchmark_func, optimizers, num_iterations, drop_rates)

if __name__ == "__main__":
    main()