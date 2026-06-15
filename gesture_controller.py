import sys
import os
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Import classifier
from sklearn.neighbors import KNeighborsClassifier

try:
    import cv2
except ImportError:
    print("Error: 'opencv-python' (cv2) is not installed.")
    sys.exit(1)

# Path to the local DINOv3 repository directory
REPO_DIR = "/Users/loganchoi/Desktop/dinov3/dinov3"
sys.path.append(REPO_DIR)

def trigger_mac_action(class_idx):
    """Executes native macOS actions using AppleScript."""
    import subprocess
    if class_idx == 0:
        # Simulate spacebar keypress (toggles Play/Pause in most video/music players)
        print("Action: Play/Pause (Spacebar)")
        subprocess.run(["osascript", "-e", 'tell application "System Events" to key code 49'])
    elif class_idx == 1:
        # Increase system volume by 10%
        print("Action: Volume Up")
        subprocess.run(["osascript", "-e", 'set volume output volume (output volume of (get volume settings) + 10)'])
    elif class_idx == 2:
        # Decrease system volume by 10%
        print("Action: Volume Down")
        subprocess.run(["osascript", "-e", 'set volume output volume (output volume of (get volume settings) - 10)'])

def main():
    # 1. Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU acceleration (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 2. Load DINOv3 backbone model
    print("Loading DINOv3 ViT-S/16 weights...")
    model = torch.hub.load(
        REPO_DIR, 
        'dinov3_vits16', 
        source='local', 
        weights='https://huggingface.co/jaychempan/dinov3/resolve/main/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
    )
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # 3. Setup transforms
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

    # Data collection arrays
    X_train = []
    y_train = []
    
    is_trained = False
    classifier = None

    # Load saved classifier and training data if it exists
    model_path = "gesture_classifier.pkl"
    if os.path.exists(model_path):
        try:
            import pickle
            with open(model_path, "rb") as f:
                saved_data = pickle.load(f)
                X_train = saved_data["X_train"]
                y_train = saved_data["y_train"]
                classifier = saved_data["classifier"]
                is_trained = True
            print(f"Loaded existing classifier '{model_path}' with {len(X_train)} samples!")
        except Exception as e:
            print(f"Error loading '{model_path}', starting fresh: {e}")

    class_names = {
        0: "Play/Pause (Open Palm)",
        1: "Volume Up (Thumbs Up)",
        2: "Volume Down (Thumbs Down)",
        3: "Neutral (No Gesture)"
    }

    # Recording states
    recording_class = None
    recording_frames_left = 0

    # Cooldown states to prevent double triggering
    last_trigger_time = 0
    cooldown_period = 1.2 # seconds between actions
    
    # Overlay notification trackers
    action_text = ""
    action_text_expiry = 0

    print("\n" + "="*60)
    print("GESTURE CONTROLLER INSTRUCTIONS:")
    print("1. Show an OPEN PALM and press '0' (records 20 frames).")
    print("2. Show a THUMBS UP and press '1' (records 20 frames).")
    print("3. Show a THUMBS DOWN and press '2' (records 20 frames).")
    print("4. Keep frame empty/natural and press '3' (records 20 frames).")
    print("5. Press 't' to train the model once you have collected data.")
    print("6. Press 'c' to clear and reset.")
    print("7. Press 'q' to quit.")
    print("="*60 + "\n")

    cv2.namedWindow("DINOv3 Teachable Gesture Controller", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Center crop to square
        h, w, _ = frame.shape
        crop_size = min(h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        cropped_frame = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]
        display_frame = cv2.resize(cropped_frame, (448, 448))

        # Preprocess PIL image for model
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        # Extract features (frozen DINOv3)
        with torch.no_grad():
            features = model.forward_features(input_tensor)
            # Use global image classification embedding
            embedding = features["x_norm_clstoken"].squeeze(0).cpu().numpy()

        # Handle data collection
        if recording_frames_left > 0:
            X_train.append(embedding)
            y_train.append(recording_class)
            recording_frames_left -= 1
            if recording_frames_left == 0:
                print(f"Finished collecting for class {recording_class}! Total samples: {y_train.count(recording_class)}")
                recording_class = None

        # Build overlay frame
        overlay = display_frame.copy()

        # Render status lines
        mode_text = "Mode: CONTROL ACTIVE" if is_trained else "Mode: COLLECTING GESTURES"
        cv2.putText(overlay, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        c0_count = y_train.count(0)
        c1_count = y_train.count(1)
        c2_count = y_train.count(2)
        c3_count = y_train.count(3)

        cv2.putText(overlay, f"0 (Open Palm)  Samples: {c0_count}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(overlay, f"1 (Thumbs Up)  Samples: {c1_count}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(overlay, f"2 (Thumbs Dn)  Samples: {c2_count}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(overlay, f"3 (Neutral)    Samples: {c3_count}", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Highlight current recording state
        if recording_class is not None:
            cv2.putText(overlay, f"RECORDING CLASS {recording_class}... ({recording_frames_left} left)", 
                        (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # Run predictions & trigger actions
        current_time = time.time()
        if is_trained and classifier is not None:
            pred_class = classifier.predict([embedding])[0]
            pred_probs = classifier.predict_proba([embedding])[0]
            confidence = pred_probs[pred_class] * 100

            pred_name = class_names[pred_class]
            
            # Predict color coding
            color = (0, 255, 0) if pred_class != 3 else (255, 120, 0)

            # Trigger action if confidence is high, class is not Neutral (3), and cooldown has elapsed
            if pred_class != 3 and confidence > 85.0:
                time_since_last = current_time - last_trigger_time
                if time_since_last > cooldown_period:
                    # Execute action
                    trigger_mac_action(pred_class)
                    last_trigger_time = current_time
                    action_text = f"TRIGGERED: {pred_name.split(' (')[0].upper()}!"
                    action_text_expiry = 20 # Keep on screen for 20 frames

            # Draw prediction bar
            cv2.rectangle(overlay, (5, 380), (443, 443), (0, 0, 0), -1)
            cv2.putText(overlay, f"GESTURE: {pred_name}", (15, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(overlay, f"CONFIDENCE: {confidence:.1f}%", (15, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Render triggered action banner overlay
        if action_text_expiry > 0:
            cv2.rectangle(overlay, (10, 220), (438, 270), (0, 180, 0), -1)
            cv2.putText(overlay, action_text, (25, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            action_text_expiry -= 1

        cv2.imshow("DINOv3 Teachable Gesture Controller", overlay)

        # Keyboard check
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # Quit
            break
        elif key == ord('0'):
            recording_class = 0
            recording_frames_left = 20
            print("Recording Open Palm...")
        elif key == ord('1'):
            recording_class = 1
            recording_frames_left = 20
            print("Recording Thumbs Up...")
        elif key == ord('2'):
            recording_class = 2
            recording_frames_left = 20
            print("Recording Thumbs Down...")
        elif key == ord('3'):
            recording_class = 3
            recording_frames_left = 20
            print("Recording Neutral...")
        elif key == ord('t'): # Train
            if len(X_train) < 4 or len(set(y_train)) < 2:
                print("Error: Collect samples for at least 2 classes before training.")
            else:
                print(f"Training KNN on {len(X_train)} samples...")
                k_val = min(3, len(X_train))
                classifier = KNeighborsClassifier(n_neighbors=k_val)
                classifier.fit(X_train, y_train)
                is_trained = True
                print("Training complete! Control mode active.")
                # Save model to disk
                try:
                    import pickle
                    with open(model_path, "wb") as f:
                        pickle.dump({
                            "X_train": X_train,
                            "y_train": y_train,
                            "classifier": classifier
                        }, f)
                    print(f"Saved classifier to '{model_path}'!")
                except Exception as e:
                    print(f"Error saving classifier: {e}")
        elif key == ord('c'): # Clear
            X_train = []
            y_train = []
            is_trained = False
            classifier = None
            recording_class = None
            recording_frames_left = 0
            action_text = ""
            action_text_expiry = 0
            # Remove saved model file
            if os.path.exists(model_path):
                try:
                    os.remove(model_path)
                    print(f"Deleted saved classifier file '{model_path}'")
                except Exception as e:
                    print(f"Error deleting file: {e}")
            print("Classifier reset. Ready to collect data.")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Gesture controller closed.")

if __name__ == "__main__":
    main()
