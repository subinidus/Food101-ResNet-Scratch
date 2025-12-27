import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Append parent directory to path to import 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import ResNetScratch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_with_svm(model, train_loader, test_loader, device):
    """
    Evaluates the model by using it as a Feature Extractor 
    and training an SVM Classifier on top of the extracted features.
    """
    model.eval()
    features = {}
    
    # Register forward hook to extract features from the Global Average Pooling layer
    def hook(m, i, o):
        features['feats'] = o.detach()
    handle = model.avgpool.register_forward_hook(hook)

    print(">>> Extracting Features...")
    
    # Extract Train Features
    train_feats, train_labels = [], []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            _ = model(inputs)
            # Flatten features
            feats = features['feats'].view(inputs.size(0), -1).cpu().numpy()
            train_feats.append(feats)
            train_labels.append(labels.numpy())
            
            # Limit samples for quick analysis
            if i >= 100: break 
            
    # Extract Test Features
    test_feats, test_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            feats = features['feats'].view(inputs.size(0), -1).cpu().numpy()
            test_feats.append(feats)
            test_labels.append(labels.numpy())

    handle.remove() # Remove hook after use
    
    # Stack features into numpy arrays
    X_train = np.vstack(train_feats)
    y_train = np.concatenate(train_labels)
    X_test = np.vstack(test_feats)
    y_test = np.concatenate(test_labels)

    print(f">>> Training SVM Classifier (Train Samples: {len(X_train)})...")
    svm = LinearSVC(max_iter=1000, C=1.0, dual=False)
    svm.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, svm.predict(X_test)) * 100
    print(f"SVM Classifier Accuracy: {acc:.2f}%")
    return acc

def main_analysis():
    # 1. Load Pre-trained Model
    model = ResNetScratch(num_classes=101).to(DEVICE)
    model_path = "../resnet18_scratch_food101_final.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Pre-trained Custom Model Loaded")
    else:
        print("Warning: Model file not found. Running with random weights for testing.")

    # 2. Prepare Data Loaders
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    data_root = '../data' 
    train_ds = torchvision.datasets.Food101(root=data_root, split='train', download=True, transform=transform)
    test_ds = torchvision.datasets.Food101(root=data_root, split='test', download=True, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # 3. Run Evaluation
    evaluate_with_svm(model, train_loader, test_loader, DEVICE)

if __name__ == "__main__":
    main_analysis()