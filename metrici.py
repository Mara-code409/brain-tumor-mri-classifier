import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_data = ImageFolder(root='brisc2025/classification_task/train', transform=transform)
test_data = ImageFolder(root='brisc2025/classification_task/test', transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Listele pentru grafice
train_losses = []
train_accuracies = []

for epoch in range(5):
    model.train()
    total_loss = 0
    num_batches = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = total_loss / num_batches
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    print(f"Epoch {epoch+1}/5 - Loss: {epoch_loss:.4f} - Accuracy train: {epoch_acc:.2f}%")

# Evaluare pe test
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        output = model(images)
        _, predicted = torch.max(output, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

clase = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print(f"\nAccuracy test: {accuracy:.2f}%")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=clase))

# Grafic 1 - Loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, 6), train_losses, marker='o', color='blue')
plt.title('Loss pe epoci')
plt.xlabel('Epoca')
plt.ylabel('Loss')
plt.grid(True)

# Grafic 2 - Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, 6), train_accuracies, marker='o', color='green')
plt.title('Accuracy pe epoci')
plt.xlabel('Epoca')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Grafic 3 - Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=clase, yticklabels=clase)
plt.title('Confusion Matrix')
plt.ylabel('Clasa reala')
plt.xlabel('Clasa prezisa')
plt.tight_layout()
plt.show()