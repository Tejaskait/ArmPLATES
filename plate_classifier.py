# plate_classifier.py
# Small CNN: input = 64x64 RGB crop (from an OpenCV candidate box),
# output = which plate (0-3) it is. This is the actual "neural network"
# object detector - cv_detector.py proposes regions, this classifies them.
#
# Reduced from 8 classes to 4 to match the single-arm, 4-plate scope.
#
# Also runnable standalone to train:  python plate_classifier.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_PLATES = 4
IMG_SIZE = 64
MODEL_PATH = "models/plate_classifier.pt"


class PlateCNN(nn.Module):
    def __init__(self, num_classes=NUM_PLATES):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 64 -> 32 -> 16 -> 8 after three pools
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # raw logits


def load_model(path=MODEL_PATH, device="cpu"):
    model = PlateCNN()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model.to(device)


def predict(model, crop_rgb_uint8, device="cpu"):
    """crop_rgb_uint8: HxWx3 numpy array, any size, RGB order.
    Returns (predicted_plate_index, confidence)."""
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    x = tf(crop_rgb_uint8).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
    return int(pred.item()), float(conf.item())


def train(data_dir="data/plate_dataset", epochs=15, batch_size=32, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    # ImageFolder assigns class indices alphabetically by whichever folders
    # are PRESENT at load time - not by folder name value. If a class
    # folder is empty and we silently delete it, every class folder after
    # it shifts down one index (e.g. delete "1", and "2" becomes index 1
    # instead of 2). The saved model's output indices would then no
    # longer match config.PLATES indices, and cv_detector.py / rl_env.py
    # would silently misattribute every detection from that point on -
    # exactly the kind of bug where a plate looks "undetectable" for no
    # visible reason. So: fail loudly instead of auto-dropping.
    all_dirs = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    empty = [d for d in all_dirs if not os.listdir(os.path.join(data_dir, d))]
    if empty:
        raise RuntimeError(
            f"Class folder(s) {empty} in {data_dir} have zero crops. Training on the "
            f"remaining folders would silently shift every class index after the gap and "
            f"desync the model's outputs from config.PLATES. Re-run dataset_capture.py "
            f"(widen random_camera_pose()'s radius range for the missing plate size) "
            f"until every class folder 0..{len(all_dirs)-1} has crops, then retrain."
        )

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=tf)
    print(f"Loaded {len(dataset)} crops across {len(dataset.classes)} classes: {dataset.classes}")

    from config import PLATES
    if len(dataset.classes) != len(PLATES) or dataset.classes != [str(i) for i in range(len(PLATES))]:
        raise RuntimeError(
            f"Dataset classes {dataset.classes} don't line up 1:1 with config.PLATES "
            f"(expected {[str(i) for i in range(len(PLATES))]}). Training now would "
            f"produce a model whose output indices don't match plate_index in "
            f"cv_detector.py/rl_env.py. Fix data_dir's folder structure before retraining."
        )

    n_val = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = PlateCNN(num_classes=len(dataset.classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total else 0.0
        print(f"epoch {epoch+1}/{epochs}  train_loss={total_loss/n_train:.4f}  val_acc={val_acc:.3f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved classifier to {MODEL_PATH}")


if __name__ == "__main__":
    train()