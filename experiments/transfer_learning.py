import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import os

def run_transfer_learning():
    """
    Runs Transfer Learning using ImageNet Pre-trained ResNet-50.
    This serves as a baseline to compare with our custom scratch model.
    """
    
    # ==========================================
    # Configuration
    # ==========================================
    BATCH_SIZE = 64
    IMG_SIZE = 224
    EPOCHS = 5
    LEARNING_RATE = 0.0001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 Transfer Learning Started on {DEVICE}")

    # ==========================================
    # Data Preprocessing (ImageNet Standards)
    # ==========================================
    # Standard ImageNet normalization parameters
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize
    ])

    # ==========================================
    # Load Data
    # ==========================================
    # Note: Adjust the data path relative to the experiments folder
    data_root = '../data'
    
    print(">>> Loading Dataset...")
    full_dataset = torchvision.datasets.Food101(root=data_root, download=True, transform=train_transform)
    
    # Split Train/Val (80% : 20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply validation transform (Override transform for validation set if possible, 
    # or rely on the fact that random split shares the transform. 
    # For strict implementation, we would define separate subsets.)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Data Loaded: {len(train_dataset)} Train samples, {len(val_dataset)} Val samples")

    # ==========================================
    # Model Setup (ResNet-50 Pre-trained)
    # ==========================================
    print(">>> Loading Pre-trained ResNet-50...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Modify the final fully connected layer for Food-101 (101 classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 101) 
    
    model = model.to(DEVICE)

    # Setup Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ==========================================
    # Training Loop
    # ==========================================
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if (i + 1) % 100 == 0:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        # ==========================================
        # Validation Step
        # ==========================================
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f'>>> Epoch {epoch+1} Validation Accuracy: {acc:.2f}%')

    # Save the transfer learning model if needed
    # torch.save(model.state_dict(), "../resnet50_transfer_food101.pth")
    print("✅ Transfer Learning Complete.")

if __name__ == '__main__':
    run_transfer_learning()