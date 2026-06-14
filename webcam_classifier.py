import sys
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Import scikit-learn classifier
from sklearn.neighbors import KNeighborsClassifier

try:
    import cv2
except ImportError:
    print("Error: 'opencv-python' (cv2) is not installed.")
    sys.exit(1)

# Path to the local DINOv3 repository directory
REPO_DIR = "/Users/loganchoi/Desktop/dinov3/dinov3"
sys.path.append(REPO_DIR)

def main():
    # 1. Device configuration (use MPS on Apple Silicon, CUDA on NVIDIA, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU acceleration (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 2. Load DINOv3 ViT-S/16 frozen backbone
    print("Loading DINOv3 ViT-S/16 model weights...")
    model = torch.hub.load(
        REPO_DIR, 
        'dinov3_vits16', 
        source='local', 
        weights='https://huggingface.co/jaychempan/dinov3/resolve/main/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
    )
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # 3. Setup preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 4. Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Data collection containers
    X_train = [] # Stored embeddings
    y_train = [] # Stored class labels (0, 1, 2)

    class_names = {
        0: "Class 0 (Phone)",
        1: "Class 1 (Mug/Cup)",
        2: "Class 2 (Neutral)"
    }

    # Training state flags
    is_trained = False
    classifier = None
    
    # Active recording state
    recording_class = None
    recording_frames_left = 0

    print("\n" + "="*50)
    # Print instructions to the console
    print("TEACHABLE MACHINE CLASSIFIER INSTRUCTIONS:")
    print("1. Hold up object A (e.g. Phone) and press '0' to collect samples.")
    print("2. Hold up object B (e.g. Mug) and press '1' to collect samples.")
    print("3. Sit normally (Neutral) and press '2' to collect samples.")
    print("4. Press 't' to train the classifier once you have collected data.")
    print("5. Press 'c' to clear and reset the classifier.")
    print("6. Press 'q' to quit.")
    print("="*50 + "\n")

    cv2.namedWindow("DINOv3 Live Teachable Classifier", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Crop to square center
        h, w, _ = frame.shape
        crop_size = min(h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        cropped_frame = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]
        
        # We'll display at 448x448
        display_frame = cv2.resize(cropped_frame, (448, 448))

        # Convert to PIL and preprocess for model
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        # Extract features (Frozen)
        with torch.no_grad():
            features = model.forward_features(input_tensor)
            # Global embedding vector (shape: [384])
            embedding = features["x_norm_clstoken"].squeeze(0).cpu().numpy()

        # Handle active recording of frames
        if recording_frames_left > 0:
            X_train.append(embedding)
            y_train.append(recording_class)
            recording_frames_left -= 1
            if recording_frames_left == 0:
                print(f"Finished collecting for {class_names[recording_class]}! Total samples: {y_train.count(recording_class)}")
                recording_class = None

        # Build status overlay
        overlay = display_frame.copy()
        
        # Show mode
        mode_text = "Mode: Predict" if is_trained else "Mode: Collect Data"
        cv2.putText(overlay, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show collection counters
        c0_count = y_train.count(0)
        c1_count = y_train.count(1)
        c2_count = y_train.count(2)
        cv2.putText(overlay, f"0: Phone Samples: {c0_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay, f"1: Mug Samples:   {c1_count}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay, f"2: Neutral Samples: {c2_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Highlight if currently recording
        if recording_class is not None:
            cv2.putText(overlay, f"RECORDING {recording_class}... ({recording_frames_left} frames left)", 
                        (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show predictions if trained
        if is_trained and classifier is not None:
            # Predict class and probability
            pred_class = classifier.predict([embedding])[0]
            pred_probs = classifier.predict_proba([embedding])[0]
            confidence = pred_probs[pred_class] * 100

            pred_name = class_names[pred_class]
            color = (0, 255, 0) if pred_class != 2 else (255, 100, 0)
            
            # Draw prediction text box
            cv2.rectangle(overlay, (5, 380), (443, 443), (0, 0, 0), -1)
            cv2.putText(overlay, f"PREDICTED: {pred_name}", (15, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(overlay, f"CONFIDENCE: {confidence:.1f}%", (15, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Render overlays
        cv2.imshow("DINOv3 Live Teachable Classifier", overlay)

        # Keyboard checks
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'): # Quit
            break
        elif key == ord('0'): # Collect Class 0
            recording_class = 0
            recording_frames_left = 15
            print(f"Collecting 15 samples for {class_names[0]}...")
        elif key == ord('1'): # Collect Class 1
            recording_class = 1
            recording_frames_left = 15
            print(f"Collecting 15 samples for {class_names[1]}...")
        elif key == ord('2'): # Collect Class 2
            recording_class = 2
            recording_frames_left = 15
            print(f"Collecting 15 samples for {class_names[2]}...")
        elif key == ord('t'): # Train
            if len(X_train) < 3 or len(set(y_train)) < 2:
                print("Error: Collect samples for at least 2 different classes before training.")
            else:
                print(f"Training KNN classifier on {len(X_train)} samples...")
                # Using simple K-Nearest Neighbors classifier
                # We use K=3 (or fewer if we don't have enough samples)
                k_val = min(3, len(X_train))
                classifier = KNeighborsClassifier(n_neighbors=k_val)
                classifier.fit(X_train, y_train)
                is_trained = True
                print("Classifier training complete! Mode switched to Predict.")
        elif key == ord('c'): # Clear
            X_train = []
            y_train = []
            is_trained = False
            classifier = None
            recording_class = None
            recording_frames_left = 0
            print("Classifier and training data cleared. Mode switched to Collect.")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Classifier pipeline closed.")

if __name__ == "__main__":
    main()
