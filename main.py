"""
Load pretrained ViT and train on CIFAR-100 for classification.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm


def get_transforms():
    """CIFAR-100 data augmentation; ViT requires 224x224 input."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=28),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return train_transform, test_transform


def load_model(num_classes=100, freeze_backbone=True):
    """Load pretrained ViT and replace classification head."""
    # Load pretrained ViT-Base (ImageNet) via timm
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=True,
        num_classes=num_classes,
    )
    if freeze_backbone:
        # Freeze backbone, only train head (classifier)
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Training")

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")

    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Hyperparameters
    batch_size = 64
    epochs = 100
    lr = 1e-4

    # Load data
    train_transform, test_transform = get_transforms()
    train_dataset = datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    # Load pretrained ViT (freeze backbone, train classifier only)
    print("Loading pretrained ViT model...")
    model = load_model(num_classes=100, freeze_backbone=True)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    criterion = nn.CrossEntropyLoss()
    # Optimize only trainable params (head)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    # Training
    print("\nStarting training...")
    best_acc = 0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "vit_cifar100_best.pt")
            print(f"  -> Saved best model (acc: {best_acc:.2f}%)")

    print(f"\nTraining complete! Best test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
