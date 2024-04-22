import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from dropgrad import DropGrad
from vit_model import vit_base_patch16_224

# Check if MPS (Metal Performance Shaders) is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders) device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (GPU) device")
else:
    device = torch.device("cpu")
    print("Using CPU device (MPS and CUDA not available)")

def train(model, optimizer, criterion, train_loader, test_loader, epochs, device):
    train_losses = []
    test_losses = []

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_total = 0

        print(f"Epoch [{epoch+1}/{epochs}]")
        print("Training...")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            train_total += images.size(0)

            if (batch_idx + 1) % 100 == 0:
                print(f"Batch [{batch_idx+1}/{len(train_loader)}] - Train Loss: {loss.item():.4f}")

        train_loss = train_loss / train_total
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0.0
        test_total = 0

        print("Evaluating...")

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                test_total += images.size(0)

                if (batch_idx + 1) % 100 == 0:
                    print(f"Batch [{batch_idx+1}/{len(test_loader)}] - Test Loss: {loss.item():.4f}")

        test_loss = test_loss / test_total
        test_losses.append(test_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        print("--" * 20)

    return train_losses, test_losses

def main():
    # Define data transforms
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # Use a smaller subset for faster experimentation
    train_subset = Subset(train_dataset, range(10000))
    test_subset = Subset(test_dataset, range(1000))

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=4)

    # Define the scenarios
    scenarios = [
        {"name": "Baseline", "dropout_rate": 0.0, "dropgrad_rate": 0.0},
        {"name": "Dropout", "dropout_rate": 0.1, "dropgrad_rate": 0.0},
        {"name": "DropGrad", "dropout_rate": 0.0, "dropgrad_rate": 0.1},
        {"name": "Dropout+DropGrad", "dropout_rate": 0.1, "dropgrad_rate": 0.1},
    ]

    # Define the optimizers
    optimizers = [
        optim.Adam,
        optim.AdamW,
        optim.SGD,
        optim.Adagrad,
        optim.Adadelta,
    ]

    # Hyperparameter grid search
    dropout_rates = [0.0, 0.1]
    dropgrad_rates = [0.0, 0.1]

    # Define the number of epochs
    epochs = 10

    try:
        # Perform grid search for each scenario and optimizer
        for scenario in scenarios:
            print(f"Scenario: {scenario['name']}")

            best_dropout_rate = 0.0
            best_dropgrad_rate = 0.0
            best_loss = float("inf")

            for dropout_rate in dropout_rates:
                for dropgrad_rate in dropgrad_rates:
                    print(f"Dropout Rate: {dropout_rate}, DropGrad Rate: {dropgrad_rate}")

                    model = vit_base_patch16_224(n_classes=10, dropout_rate=dropout_rate, patch_size=32, embed_dim=256, depth=8, num_heads=8)
                    model.to(device)
                    criterion = nn.CrossEntropyLoss()

                    for optimizer_class in optimizers:
                        print(f"Optimizer: {optimizer_class.__name__}")

                        base_optimizer = optimizer_class(model.parameters(), lr=0.001)
                        optimizer = DropGrad(base_optimizer, drop_rate=dropgrad_rate)

                        train_losses, test_losses = train(model, optimizer, criterion, train_loader, test_loader, epochs, device)

                        if test_losses[-1] < best_loss:
                            best_dropout_rate = dropout_rate
                            best_dropgrad_rate = dropgrad_rate
                            best_loss = test_losses[-1]

            print(f"Best Dropout Rate: {best_dropout_rate}, Best DropGrad Rate: {best_dropgrad_rate}")
            print("--" * 20)

            # Train the model with the best hyperparameters for each scenario and optimizer
            for optimizer_class in optimizers:
                print(f"Training with {optimizer_class.__name__} optimizer")

                model = vit_base_patch16_224(n_classes=10, dropout_rate=best_dropout_rate, patch_size=32, embed_dim=256, depth=8, num_heads=8)
                model.to(device)
                criterion = nn.CrossEntropyLoss()
                base_optimizer = optimizer_class(model.parameters(), lr=0.001)
                optimizer = DropGrad(base_optimizer, drop_rate=best_dropgrad_rate)

                train_losses, test_losses = train(model, optimizer, criterion, train_loader, test_loader, epochs, device)

                # Save the loss values for visualization
                torch.save({"train_losses": train_losses, "test_losses": test_losses},
                           f"losses_{scenario['name']}_{optimizer_class.__name__}.pth")

                print("--" * 20)

    except KeyboardInterrupt:
        print("Training interrupted by user.")

    finally:
        # Save the loss values for visualization if available
        for scenario in scenarios:
            for optimizer_class in optimizers:
                try:
                    torch.save({"train_losses": train_losses, "test_losses": test_losses},
                               f"losses_{scenario['name']}_{optimizer_class.__name__}.pth")
                except NameError:
                    pass

if __name__ == "__main__":
    main()