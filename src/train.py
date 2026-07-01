import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image

from dataset import CatDogDataset, data_transforms
from model import get_model


def get_clean_paths_and_labels(data_dir):
    categories = ['Cat', 'Dog']
    file_paths = []
    labels = []

    print("Scanning and verifying image integrity....")

    for label_idx, category in enumerate(categories):
        category_dir = os.path.join(data_dir, category)
        # Match jpg, jpeg, and png variations
        raw_paths = glob.glob(os.path.join(category_dir, "*.[jJ][pP]*[gG]")) + \
                    glob.glob(os.path.join(category_dir, "*.[pP][nN][gG]"))

        for path in raw_paths:
            try:
                with Image.open(path) as img:
                    img.verify()  # Check if file is corrupted
                file_paths.append(path)
                labels.append(label_idx)
            except Exception:
                continue  # Skip broken files

    return file_paths, labels


def train_pipeline():
    DATA_DIR = "../data/raw"
    BATCH_SIZE = 32
    EPOCHS = 2

    paths, labels = get_clean_paths_and_labels(DATA_DIR) # prepare path

    # train val split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.20, random_state=42, stratify=labels
    )

    # instantiate dataset and data loaders
    train_dataset = CatDogDataset(train_paths, train_labels, transform=data_transforms)
    val_dataset = CatDogDataset(val_paths, val_labels, transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # initialise models
    model = get_model(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    print(f"\nReady to train! Data split complete.")
    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    # Core Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Batch [{batch_idx + 1}/{len(train_loader)}] | Avg Loss: {running_loss / (batch_idx + 1):.4f}")

        # Evaluation Loop (Validation)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = (correct / total) * 100
        print(f"🎯 Epoch {epoch + 1} Validation Accuracy: {accuracy:.2f}%")

    # Export the trained brain weights
    os.makedirs("../weights", exist_ok=True)
    torch.save(model.state_dict(), "../weights/model.pth")
    print("Model architecture weights saved successfully to '../weights/model.pth'!")


if __name__ == "__main__":
    train_pipeline()