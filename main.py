# =========================
# SSL FIX (for macOS)
# =========================
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# =========================
# IMPORTS
# =========================
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix

from models.cnn import CNN
from train.train import train, test
from utils.quantization import quantize_tensor

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# =========================
# DATA (Overfitting Fix)
# =========================
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=train_transform
)

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=test_transform
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# =========================
# MODEL
# =========================
model = CNN().to(device)
criterion = torch.nn.CrossEntropyLoss()

# Weight decay added (regularization)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# =========================
# TRAIN FULL MODEL
# =========================
print("\nTraining Full Model...")
full_curve = train(model, trainloader, criterion, optimizer, device, epochs=8)

acc_fp = test(model, testloader, device, detailed=True)
print("Full Precision Accuracy:", acc_fp)

# =========================
# QUANTIZATION TEST
# =========================
bits_list = [2, 3, 4]
direct_results = []
gradual_results = []

# =========================
# DIRECT QUANTIZATION
# =========================
for bits in bits_list:
    model_d = CNN().to(device)
    model_d.load_state_dict(model.state_dict())

    for p in model_d.parameters():
        p.data = quantize_tensor(p.data, bits)

    acc = test(model_d, testloader, device)
    direct_results.append(acc)
    print(f"Direct {bits}-bit:", acc)

# =========================
# GRADUAL QUANTIZATION
# =========================
gradual_curve = []

for bits in bits_list:
    model_g = CNN().to(device)
    model_g.load_state_dict(model.state_dict())

    optimizer = torch.optim.Adam(model_g.parameters(), lr=0.0005, weight_decay=1e-4)

    layers = list(model_g.parameters())

    for i in range(len(layers)):
        layers[i].data = quantize_tensor(layers[i].data, bits)
        curve = train(model_g, trainloader, criterion, optimizer, device, epochs=1)
        gradual_curve.extend(curve)

    acc = test(model_g, testloader, device)
    gradual_results.append(acc)
    print(f"Gradual {bits}-bit:", acc)

# =========================
# GRAPH 1: BIT vs ACCURACY
# =========================
plt.figure()

plt.plot(bits_list, direct_results, marker='o', label='Direct')
plt.plot(bits_list, gradual_results, marker='o', label='Gradual')
plt.axhline(y=acc_fp, linestyle='--', label='Full Precision')

plt.xlabel("Bit-width")
plt.ylabel("Accuracy (%)")
plt.title("Direct vs Gradual Quantization")
plt.legend()
plt.grid()

plt.savefig("results/graphs/bit_vs_accuracy.png")

# =========================
# GRAPH 2: TRAINING CURVE
# =========================
plt.figure()

plt.plot(full_curve, marker='o', label='Full Training')
plt.plot(gradual_curve, marker='o', label='Gradual')

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy vs Epochs")
plt.legend()
plt.grid()

plt.savefig("results/graphs/training_curve.png")

# =========================
# CONFUSION MATRIX
# =========================
def plot_confusion_matrix(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=False, fmt='d')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("results/graphs/confusion_matrix.png")
    plt.show()

plot_confusion_matrix(model, testloader, device)

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "results/saved_models/cnn_fp.pth")