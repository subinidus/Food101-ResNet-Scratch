import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from src.model import ResNetScratch
from src.transforms import get_transforms
import os

# ==========================================
# Configuration & Hyperparameters
# ==========================================
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STAGE1_EPOCHS = 80
STAGE2_EPOCHS = 20

def train_one_epoch(model, loader, criterion, optimizer, epoch, stage_name):
    """
    Runs training for one epoch.
    """
    model.train()
    running_loss = 0.0
    total_batches = len(loader)
    
    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i + 1) % 100 == 0:
            print(f"[{stage_name}] Epoch {epoch+1}/{STAGE1_EPOCHS if stage_name=='Stage1' else STAGE2_EPOCHS}, Batch {i+1}/{total_batches}: Loss {running_loss/100:.4f}")
            running_loss = 0.0

def main():
    print(f" Training Started on {DEVICE}")
    
    # 1. Load Dataset & Apply Transforms
    transform_s1, transform_s2, transform_test = get_transforms()
    
    data_root = './data'
    
    # Define Datasets
    train_ds_s1 = torchvision.datasets.Food101(root=data_root, split='train', download=True, transform=transform_s1)
    train_ds_s2 = torchvision.datasets.Food101(root=data_root, split='train', download=True, transform=transform_s2)
    # test_ds = torchvision.datasets.Food101(root=data_root, split='test', download=True, transform=transform_test)

    train_loader_1 = DataLoader(train_ds_s1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    train_loader_2 = DataLoader(train_ds_s2, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 2. Initialize Model
    model = ResNetScratch(num_classes=101).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ==========================================
    # Stage 1: Representation Learning (Strong Augmentation)
    # ==========================================
    print(f"\n>>> [Stage 1] Start Training (Epoch 1~{STAGE1_EPOCHS}) - Strong Augmentation")
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE1_EPOCHS)

    for epoch in range(STAGE1_EPOCHS):
        train_one_epoch(model, train_loader_1, criterion, optimizer, epoch, "Stage1")
        scheduler.step()
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"resnet_stage1_epoch{epoch+1}.pth")

    # ==========================================
    # Stage 2: Fine-tuning (Weak Augmentation)
    # ==========================================
    print(f"\n>>> [Stage 2] Start Fine-tuning (Epoch {STAGE1_EPOCHS+1}~{STAGE1_EPOCHS+STAGE2_EPOCHS}) - Weak Augmentation")
    
    # Decrease learning rate for stable convergence
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE2_EPOCHS)

    for epoch in range(STAGE2_EPOCHS):
        train_one_epoch(model, train_loader_2, criterion, optimizer, epoch, "Stage2")
        scheduler.step()

    # Save Final Model
    torch.save(model.state_dict(), "resnet18_scratch_food101_final.pth")
    print("\n Training Complete. Model Saved.")

if __name__ == '__main__':
    main()