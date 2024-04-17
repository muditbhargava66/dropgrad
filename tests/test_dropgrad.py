import torch
from torch.optim import Adam
from dropgrad import DropGrad

def test_dropgrad_initialization():
    net = torch.nn.Linear(10, 20)
    optimizer = Adam(net.parameters(), lr=0.001)
    drop_rate = 0.1
    dropgrad_optimizer = DropGrad(optimizer, drop_rate=drop_rate)
    assert dropgrad_optimizer.optimizer == optimizer
    assert dropgrad_optimizer.drop_rate == drop_rate

def test_dropgrad_step():
    net = torch.nn.Linear(10, 20)
    optimizer = Adam(net.parameters(), lr=0.001)
    drop_rate = 0.5
    dropgrad_optimizer = DropGrad(optimizer, drop_rate=drop_rate)

    input_data = torch.randn(5, 10)
    target = torch.randn(5, 20)

    # Forward pass
    output = net(input_data)
    loss = torch.nn.functional.mse_loss(output, target)

    # Backward pass
    loss.backward()

    # Optimize with DropGrad
    dropgrad_optimizer.step()

    for param in net.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()

def test_dropgrad_zero_grad():
    net = torch.nn.Linear(10, 20)
    optimizer = Adam(net.parameters(), lr=0.001)
    drop_rate = 0.5
    dropgrad_optimizer = DropGrad(optimizer, drop_rate=drop_rate)

    input_data = torch.randn(5, 10)
    target = torch.randn(5, 20)

    # Forward and backward pass
    output = net(input_data)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    # Check gradients before zero_grad
    for param in net.parameters():
        assert param.grad is not None

    # Zero gradients
    dropgrad_optimizer.zero_grad()

    # Check gradients after zero_grad
    for param in net.parameters():
        assert param.grad is None or torch.all(param.grad == 0)

def test_dropgrad_full_update_drop():
    net = torch.nn.Linear(10, 20)
    optimizer = Adam(net.parameters(), lr=0.001)
    drop_rate = 1.0  # Always drop the full update
    dropgrad_optimizer = DropGrad(optimizer, drop_rate=drop_rate, full_update_drop=True)

    input_data = torch.randn(5, 10)
    target = torch.randn(5, 20)

    # Forward and backward pass
    output = net(input_data)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    # Optimize with DropGrad (full update drop)
    initial_params = [param.clone() for param in net.parameters()]
    dropgrad_optimizer.step()

    # Check if parameters remain unchanged due to full update drop
    for param, initial_param in zip(net.parameters(), initial_params):
        assert torch.allclose(param, initial_param)