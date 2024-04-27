import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.optim as optim
from lion_pytorch import Lion

from dropgrad import DropGrad
from examples.vit_experiments.vit_model import vit_base_patch16_224

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestDropGrad(unittest.TestCase):
    def setUp(self):
        self.model = vit_base_patch16_224(
            n_classes=10,
            dropout_rate=0.1,
            patch_size=32,
            embed_dim=256,
            depth=8,
            num_heads=8,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizers = [
            optim.Adam(self.model.parameters(), lr=0.001),
            optim.AdamW(self.model.parameters(), lr=0.001),
            optim.SGD(self.model.parameters(), lr=0.001),
            optim.Adagrad(self.model.parameters(), lr=0.001),
            optim.Adadelta(self.model.parameters(), lr=0.001),
            Lion(self.model.parameters(), lr=0.001),
        ]
        self.drop_rates = [0.0, 0.1, 0.2]

    def test_dropgrad_initialization(self):
        for optimizer in self.optimizers:
            dropgrad_optimizer = DropGrad(optimizer, drop_rate=0.1)
            self.assertIsInstance(dropgrad_optimizer, DropGrad)
            self.assertEqual(dropgrad_optimizer.drop_rate, 0.1)

    def test_dropgrad_step(self):
        for optimizer in self.optimizers:
            for drop_rate in self.drop_rates:
                dropgrad_optimizer = DropGrad(optimizer, drop_rate=drop_rate)

                # Generate dummy input and target
                input_data = torch.randn(32, 3, 224, 224)
                target = torch.randint(0, 10, (32,))

                # Forward pass
                output = self.model(input_data)
                loss = self.criterion(output, target)

                # Backward pass
                dropgrad_optimizer.zero_grad()
                loss.backward()
                dropgrad_optimizer.step()

                # Check if the model parameters have been updated
                for param in self.model.parameters():
                    self.assertFalse(torch.all(param.grad == 0))

    def test_dropgrad_drop_rate_scheduling(self):
        initial_drop_rate = 0.1
        final_drop_rate = 0.5
        num_steps = 100

        for optimizer in self.optimizers:
            dropgrad_optimizer = DropGrad(optimizer, drop_rate=initial_drop_rate)
            wrapped_optimizer = optim.SGD(
                optimizer.param_groups[0]["params"], lr=0.001
            )  # Pass the original optimizer's params
            scheduler = torch.optim.lr_scheduler.LinearLR(
                wrapped_optimizer,
                start_factor=initial_drop_rate,
                end_factor=final_drop_rate,
                total_iters=num_steps,
            )

            for i in range(num_steps):
                # Generate dummy input and target
                input_data = torch.randn(32, 3, 224, 224)
                target = torch.randint(0, 10, (32,))

                # Forward pass
                output = self.model(input_data)
                loss = self.criterion(output, target)

                # Backward pass
                wrapped_optimizer.zero_grad()
                loss.backward()
                wrapped_optimizer.step()
                scheduler.step()

                # Update the drop rate manually
                dropgrad_optimizer.drop_rate = (
                    initial_drop_rate
                    + (final_drop_rate - initial_drop_rate) * (i + 1) / num_steps
                )

            # Check if the drop rate has been updated
            self.assertAlmostEqual(
                dropgrad_optimizer.drop_rate, final_drop_rate, places=6
            )

    def test_loss_values_saved(self):
        scenarios = [
            {"name": "Baseline", "dropout_rate": 0.0, "dropgrad_rate": 0.0},
            {"name": "Dropout", "dropout_rate": 0.1, "dropgrad_rate": 0.0},
            {"name": "DropGrad", "dropout_rate": 0.0, "dropgrad_rate": 0.1},
            {"name": "Dropout+DropGrad", "dropout_rate": 0.1, "dropgrad_rate": 0.1},
        ]

        for scenario in scenarios:
            for optimizer_class in self.optimizers:
                pth_file = f"examples/vit_experiments/losses_{scenario['name']}_{type(optimizer_class).__name__}.pth"
                self.assertTrue(
                    os.path.isfile(pth_file),
                    f"Loss values not saved for scenario: {scenario['name']}, optimizer: {type(optimizer_class).__name__}",
                )


if __name__ == "__main__":
    unittest.main()
