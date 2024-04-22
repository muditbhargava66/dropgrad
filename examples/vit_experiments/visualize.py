import torch
import matplotlib.pyplot as plt

def main():
    scenarios = [
        {"name": "Baseline", "dropout_rate": 0.0, "dropgrad_rate": 0.0},
        {"name": "Dropout", "dropout_rate": 0.1, "dropgrad_rate": 0.0},
        {"name": "DropGrad", "dropout_rate": 0.0, "dropgrad_rate": 0.1},
        {"name": "Dropout+DropGrad", "dropout_rate": 0.1, "dropgrad_rate": 0.1},
    ]

    optimizers = [
        "Adam",
        "AdamW",
        "SGD",
        "Adagrad",
        "Adadelta",
        # "Lion",  # Uncomment if you have the Lion optimizer available
    ]

    for optimizer_name in optimizers:
        plt.figure(figsize=(10, 5))
        for scenario in scenarios:
            losses = torch.load(f"losses_{scenario['name']}_{optimizer_name}.pth")
            train_losses = losses["train_losses"]
            test_losses = losses["test_losses"]
            plt.plot(train_losses, label=f"{scenario['name']} - Train Loss")
            plt.plot(test_losses, label=f"{scenario['name']} - Test Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Train and Test Losses - {optimizer_name}")
        plt.savefig(f"loss_plot_{optimizer_name}.png")
        plt.close()

if __name__ == "__main__":
    main()