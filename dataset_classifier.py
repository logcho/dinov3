import sys
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# Path to the local DINOv3 repository
REPO_DIR = "/Users/loganchoi/Desktop/dinov3/dinov3"
sys.path.append(REPO_DIR)

def main():
    # Configure device (MPS for Mac GPU, CUDA for Nvidia, CPU otherwise)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU acceleration (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 1. Load the DINOv3 backbone model
    print("Loading DINOv3 ViT-S/16...")
    model = torch.hub.load(
        REPO_DIR, 
        'dinov3_vits16', 
        source='local', 
        weights='https://huggingface.co/jaychempan/dinov3/resolve/main/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
    )
    model = model.to(device)
    model.eval()
    print("Backbone loaded successfully!")

    # 2. Image normalization and resizing to 224x224 (required size for the model)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 3. Download CIFAR-10 train & test sets using torchvision
    print("Downloading/Loading CIFAR-10 dataset...")
    # This downloads the dataset to a local './data' folder if it is not already present
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # CIFAR-10 class names
    classes = train_dataset.classes

    # Take a subset of 1000 train images and 200 test images to make feature extraction very fast
    np.random.seed(42)
    train_indices = np.random.choice(len(train_dataset), 1000, replace=False)
    test_indices = np.random.choice(len(test_dataset), 200, replace=False)

    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

    # 4. Extract frozen embeddings for the training set
    print("\nExtracting DINOv3 embeddings for training set (1000 images)...")
    X_train, y_train = [], []
    with torch.no_grad():
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            # Extract [CLS] global token embedding
            features = model.forward_features(images)
            embeddings = features["x_norm_clstoken"].cpu().numpy()
            X_train.append(embeddings)
            y_train.append(labels.numpy())
            
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Extract frozen embeddings for the test set
    print("\nExtracting DINOv3 embeddings for test set (200 images)...")
    X_test, y_test = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            features = model.forward_features(images)
            embeddings = features["x_norm_clstoken"].cpu().numpy()
            X_test.append(embeddings)
            y_test.append(labels.numpy())

    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # 5. Train the downstream Logistic Regression classifier
    print("\nTraining Logistic Regression classifier on top of the frozen embeddings...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # 6. Evaluate predictions
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print(f"CLASSIFIER PERFORMANCE (DINOv3 ViT-S/16 + Logistic Regression)")
    print(f"Overall Test Accuracy: {accuracy:.2%}")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=classes))
    print("="*60)

    # Show a few sample predictions
    print("\nSample Predictions:")
    for idx in range(10):
        true_label = classes[y_test[idx]]
        pred_label = classes[y_pred[idx]]
        status = "✅" if true_label == pred_label else "❌"
        print(f"Image {idx:02d}: True: {true_label:<12} Predicted: {pred_label:<12} {status}")

if __name__ == "__main__":
    main()
