import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from dropgrad import DropGrad

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create an instance of the network
net = Net()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Wrap the optimizer with DropGrad
drop_rate = 0.1
dropgrad_optimizer = DropGrad(optimizer, drop_rate=drop_rate)

# Define the learning rate scheduler
lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Training loop
num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    for i in range(100):
        # Generate random input and target data
        inputs = torch.randn(batch_size, 10)
        targets = torch.randn(batch_size, 5)

        # Forward pass
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        dropgrad_optimizer.zero_grad()
        loss.backward()
        dropgrad_optimizer.step()

    # Update the learning rate
    lr_scheduler.step()

    # Print the average loss and current learning rate for every epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Learning Rate: {lr_scheduler.get_last_lr()[0]:.6f}")