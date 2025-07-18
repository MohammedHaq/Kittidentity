import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import os

# === Configuration ===
DATA_DIR = Path("model/data/cat-breeds-dataset/images")
MODEL_PATH = Path("model/saved_models/cat_resnet18.pth")
CLASSES_PATH = Path("model/saved_models/classes.txt")
BATCH_SIZE = 32
EPOCHS = 5
NUM_WORKERS = 2
LEARNING_RATE = 0.001

def main():
    # === Image Transformations ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # === Load Dataset ===
    print("📁 Loading dataset...")
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    class_names = dataset.classes
    num_classes = len(class_names)
    print(f"✅ Found {len(dataset)} images across {num_classes} classes.")

    # === Split Dataset ===
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # === Model Setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # === Training Loop ===
    print("🚀 Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"📦 Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.4f}")

    # === Save Model & Class Labels ===
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    with open(CLASSES_PATH, "w") as f:
        for cls in class_names:
            f.write(cls + "\n")

    print("✅ Training complete. Model and class names saved.")

# === Multiprocessing Safe Entry Point (required for macOS/Windows) ===
if __name__ == "__main__":
    main()
