"""
Mathematical Analysis of DropGrad's Effect on Optimizers
"""

import numpy as np
import matplotlib.pyplot as plt

def sgd_update(params, grads, lr):
    """
    Stochastic Gradient Descent (SGD) update rule.
    """
    return params - lr * grads

def adam_update(params, grads, m, v, t, lr, beta1, beta2, eps):
    """
    Adam update rule.
    """
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * (grads ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    return params - lr * m_hat / (np.sqrt(v_hat) + eps), m, v

def lion_update(params, grads, m, t, lr, beta1, beta2):
    """
    Lion update rule.
    """
    m = beta1 * m + (1 - beta1) * grads
    m_hat = m / (1 - beta1 ** t)
    update = lr * m_hat / (np.abs(m_hat) + beta2)
    return params - update, m

def dropgrad_update(params, grads, drop_rate):
    """
    DropGrad modification of the gradient update.
    """
    mask = np.random.binomial(1, 1 - drop_rate, size=grads.shape)
    return params - (grads * mask) / (1 - drop_rate)

def analyze_optimizer(optimizer, num_iterations, drop_rate=0.0):
    """
    Analyze the effect of DropGrad on an optimizer.
    """
    params = np.zeros(10)
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    trajectories = []
    for _ in range(num_iterations):
        grads = np.random.randn(*params.shape)
        if optimizer == "sgd":
            params = sgd_update(params, grads, lr)
        elif optimizer == "adam":
            params, m, v = adam_update(params, grads, m, v, _ + 1, lr, beta1, beta2, eps)
        elif optimizer == "lion":
            params, m = lion_update(params, grads, m, _ + 1, lr, beta1, beta2)

        if drop_rate > 0:
            params = dropgrad_update(params, grads, drop_rate)

        trajectories.append(params.copy())

    return np.array(trajectories)

def visualize_trajectories(optimizer, num_iterations, drop_rates):
    """
    Visualize the optimization trajectories with different drop rates.
    """
    trajectories = []
    for drop_rate in drop_rates:
        trajectories.append(analyze_optimizer(optimizer, num_iterations, drop_rate))

    plt.figure(figsize=(8, 6))
    for i, drop_rate in enumerate(drop_rates):
        plt.plot(trajectories[i][:, 0], trajectories[i][:, 1], label=f"Drop Rate: {drop_rate}")
    plt.xlabel("Parameter 1")
    plt.ylabel("Parameter 2")
    plt.title(f"Optimization Trajectories ({optimizer.upper()})")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    num_iterations = 1000
    drop_rates = [0.0, 0.1, 0.2, 0.3]

    # Analyze SGD optimizer
    visualize_trajectories("sgd", num_iterations, drop_rates)

    # Analyze Adam optimizer
    visualize_trajectories("adam", num_iterations, drop_rates)

    # Analyze Lion optimizer
    visualize_trajectories("lion", num_iterations, drop_rates)

if __name__ == "__main__":
    main()