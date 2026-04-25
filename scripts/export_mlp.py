"""Train a 784→128→ReLU→10 MLP on MNIST, verify ≥95% test accuracy,
then export weights and test set as raw float32/int32 binaries for C++."""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# ── Model ─────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 784)
        return self.fc2(torch.relu(self.fc1(x)))


# ── Data ──────────────────────────────────────────────────────────────────────

transform = transforms.Compose([transforms.ToTensor()])
mnist_root = "./data/mnist"
train_set = datasets.MNIST(mnist_root, train=True,  download=True, transform=transform)
test_set  = datasets.MNIST(mnist_root, train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=1000, shuffle=False)


# ── Train ─────────────────────────────────────────────────────────────────────

model     = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}/{EPOCHS}  loss={total_loss / len(train_loader):.4f}")


# ── Evaluate ──────────────────────────────────────────────────────────────────

model.eval()
correct = total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += len(labels)
accuracy = correct / total
print(f"Test accuracy: {accuracy:.4f}")
if accuracy < 0.95:
    print("ERROR: accuracy < 0.95; model did not converge — try increasing EPOCHS.")
    sys.exit(1)


# ── Export weights ────────────────────────────────────────────────────────────
# C++ linear forward: y = x @ W + b
# PyTorch stores weight as [out, in]; we export W.T so C++ can do x @ W directly.

os.makedirs("data", exist_ok=True)

model.fc1.weight.T.detach().numpy().astype(np.float32).tofile("data/fc1_weight.bin")
model.fc1.bias.detach().numpy().astype(np.float32).tofile("data/fc1_bias.bin")
model.fc2.weight.T.detach().numpy().astype(np.float32).tofile("data/fc2_weight.bin")
model.fc2.bias.detach().numpy().astype(np.float32).tofile("data/fc2_bias.bin")

print("Weights exported:")
print(f"  fc1_weight: {model.fc1.weight.T.shape}  → data/fc1_weight.bin")
print(f"  fc1_bias:   {model.fc1.bias.shape}       → data/fc1_bias.bin")
print(f"  fc2_weight: {model.fc2.weight.T.shape}  → data/fc2_weight.bin")
print(f"  fc2_bias:   {model.fc2.bias.shape}        → data/fc2_bias.bin")


# ── Export test set ───────────────────────────────────────────────────────────

all_images, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        all_images.append(imgs.view(-1, 784).numpy().astype(np.float32))
        all_labels.append(labels.numpy().astype(np.int32))

images_np = np.concatenate(all_images)  # [10000, 784]  float32
labels_np = np.concatenate(all_labels)  # [10000]        int32

images_np.tofile("data/mnist_test_images.bin")
labels_np.tofile("data/mnist_test_labels.bin")

print(f"Test data exported: {images_np.shape} images, {labels_np.shape} labels → data/")
print("Done.")
