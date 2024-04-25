import os
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
    ]

    for optimizer_name in optimizers:
        plt.figure(figsize=(10, 5))
        for scenario in scenarios:
            file_path = os.path.join(".", f"losses_{scenario['name']}_{optimizer_name}.pth")
            if os.path.exists(file_path):
                losses = torch.load(file_path)
                train_losses = losses["train_losses"]
                test_losses = losses["test_losses"]
                plt.plot(train_losses, label=f"{scenario['name']} - Train Loss")
                plt.plot(test_losses, label=f"{scenario['name']} - Test Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Train and Test Losses - {optimizer_name}")
        plt.tight_layout()
        
        output_dir = os.path.join(".", "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"loss_plot_{optimizer_name}.png")
        plt.savefig(output_path)
        
        plt.close()

if __name__ == "__main__":
    main()