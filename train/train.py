import torch
from sklearn.metrics import confusion_matrix, classification_report

def train(model, loader, criterion, optimizer, device, epochs=5):
    model.train()
    acc_list = []

    for epoch in range(epochs):
        correct = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        acc_list.append(acc)
        print(f"Epoch {epoch+1}: {acc:.2f}%")

    return acc_list


def test(model, loader, device, detailed=False):
    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total

    if detailed:
        print("\nClassification Report:\n")
        print(classification_report(all_labels, all_preds))

        print("\nConfusion Matrix:\n")
        print(confusion_matrix(all_labels, all_preds))

    return acc