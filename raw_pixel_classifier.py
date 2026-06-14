import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def main():
    print("Running classification WITHOUT DINOv3 (using raw pixel values)...")

    # 1. Image transform: Convert to tensor, but keep original size (32x32)
    # No upscaling or normalization needed, just raw pixels scaled to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 2. Download/Load CIFAR-10 train & test sets
    print("Downloading/Loading CIFAR-10 dataset...")
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    classes = train_dataset.classes

    # 3. Take the exact same random subset of 1000 train and 200 test images (same seed 42)
    np.random.seed(42)
    train_indices = np.random.choice(len(train_dataset), 1000, replace=False)
    test_indices = np.random.choice(len(test_dataset), 200, replace=False)

    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    # 4. Prepare raw pixel features (flatten 32x32x3 image tensors to a 3072-dimensional vector)
    print("\nPreparing raw pixel features for training set (1000 images)...")
    X_train, y_train = [], []
    for image, label in train_subset:
        # Flatten image shape [3, 32, 32] -> [3072]
        pixel_vector = image.numpy().flatten()
        X_train.append(pixel_vector)
        y_train.append(label)
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print("Preparing raw pixel features for test set (200 images)...")
    X_test, y_test = [], []
    for image, label in test_subset:
        pixel_vector = image.numpy().flatten()
        X_test.append(pixel_vector)
        y_test.append(label)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # 5. Train the downstream Logistic Regression classifier directly on raw pixels
    print("\nTraining Logistic Regression classifier directly on raw pixels...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # 6. Evaluate predictions
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print(f"CLASSIFIER PERFORMANCE (Raw Pixels + Logistic Regression)")
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
