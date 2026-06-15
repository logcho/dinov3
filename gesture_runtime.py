import sys
import os
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pickle
import subprocess

try:
    import cv2
except ImportError:
    print("Error: 'opencv-python' (cv2) is not installed.")
    sys.exit(1)

# Path to the local DINOv3 repository directory
REPO_DIR = "/Users/loganchoi/Desktop/dinov3/dinov3"
sys.path.append(REPO_DIR)

# -------------------------------------------------------------
# MODULAR GESTURE ACTION FUNCTIONS
# Connect gestures to these Python functions to customize actions
# -------------------------------------------------------------

def toggle_play_pause():
    """Toggles Play/Pause in media players via Spacebar AppleScript."""
    print("[Action Executing] Play/Pause (Spacebar)")
    subprocess.run(["osascript", "-e", 'tell application "System Events" to key code 49'])

def volume_up():
    """Increases system volume by 10% via AppleScript."""
    print("[Action Executing] Volume Up")
    subprocess.run(["osascript", "-e", 'set volume output volume (output volume of (get volume settings) + 10)'])

def volume_down():
    """Decreases system volume by 10% via AppleScript."""
    print("[Action Executing] Volume Down")
    subprocess.run(["osascript", "-e", 'set volume output volume (output volume of (get volume settings) - 10)'])

def neutral_action():
    """Neutral state action - does nothing."""
    pass

# Map class index to actual Python functions
GESTURE_ACTIONS = {
    0: toggle_play_pause,
    1: volume_up,
    2: volume_down,
    3: neutral_action
}

# Human-readable names for UI overlay
CLASS_NAMES = {
    0: "Play/Pause (Open Palm)",
    1: "Volume Up (Thumbs Up)",
    2: "Volume Down (Thumbs Down)",
    3: "Neutral (No Gesture)"
}

def main():
    model_path = "gesture_classifier.pkl"
    
    # 1. Check if model file exists
    if not os.path.exists(model_path):
        print("\n" + "="*70)
        print("ERROR: Pre-trained classifier file 'gesture_classifier.pkl' not found.")
        print("Please train your gestures first by running: python3 gesture_controller.py")
        print("="*70 + "\n")
        sys.exit(1)

    # 2. Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU acceleration (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 3. Load DINOv3 backbone model
    print("Loading DINOv3 ViT-S/16 weights...")
    try:
        model = torch.hub.load(
            REPO_DIR, 
            'dinov3_vits16', 
            source='local', 
            weights='https://huggingface.co/jaychempan/dinov3/resolve/main/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
        )
        model = model.to(device)
        model.eval()
        print("DINOv3 model loaded successfully!")
    except Exception as e:
        print(f"Error loading DINOv3 model: {e}")
        sys.exit(1)

    # 4. Load saved classifier data
    print(f"Loading pre-trained classifier '{model_path}'...")
    try:
        with open(model_path, "rb") as f:
            saved_data = pickle.load(f)
            X_train = saved_data["X_train"]
            y_train = saved_data["y_train"]
            classifier = saved_data["classifier"]
        print(f"Loaded classifier successfully with {len(X_train)} samples!")
    except Exception as e:
        print(f"Error reading classifier file: {e}")
        sys.exit(1)

    # Setup image preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Cooldown states to prevent double triggering
    last_trigger_time = 0
    cooldown_period = 1.2  # seconds between actions
    
    # Overlay notification trackers
    action_text = ""
    action_text_expiry = 0

    print("\n" + "="*60)
    print("GESTURE RUNTIME CONTROLLER ACTIVE:")
    print("- Show gestures to trigger mapped macOS actions.")
    print("- Press 'q' in the camera window to quit.")
    print("="*60 + "\n")

    cv2.namedWindow("DINOv3 Gesture Controller (Runtime)", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Horizontal flip for natural mirror effect
        frame = cv2.flip(frame, 1)

        # Center crop to a square aspect ratio
        h, w, _ = frame.shape
        crop_size = min(h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        cropped_frame = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]
        display_frame = cv2.resize(cropped_frame, (448, 448))

        # Preprocess PIL image for the DINOv3 model
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        # Extract features
        with torch.no_grad():
            features = model.forward_features(input_tensor)
            embedding = features["x_norm_clstoken"].squeeze(0).cpu().numpy()

        # Run prediction
        pred_class = classifier.predict([embedding])[0]
        pred_probs = classifier.predict_proba([embedding])[0]
        confidence = pred_probs[pred_class] * 100

        pred_name = CLASS_NAMES[pred_class]
        color = (0, 255, 0) if pred_class != 3 else (255, 120, 0)

        # Check confidence and trigger mapped action
        current_time = time.time()
        if pred_class != 3 and confidence > 85.0:
            time_since_last = current_time - last_trigger_time
            if time_since_last > cooldown_period:
                # Retrieve and execute mapped python action
                action_fn = GESTURE_ACTIONS.get(pred_class, neutral_action)
                action_fn()
                
                last_trigger_time = current_time
                action_text = f"TRIGGERED: {pred_name.split(' (')[0].upper()}!"
                action_text_expiry = 20  # Display action overlay for 20 frames

        # Build UI overlay
        overlay = display_frame.copy()
        
        # Draw status
        cv2.putText(overlay, "Runtime Mode: ACTIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        
        # Display prediction bar
        cv2.rectangle(overlay, (5, 380), (443, 443), (0, 0, 0), -1)
        cv2.putText(overlay, f"GESTURE: {pred_name}", (15, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(overlay, f"CONFIDENCE: {confidence:.1f}%", (15, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display action trigger banner
        if action_text_expiry > 0:
            cv2.rectangle(overlay, (10, 220), (438, 270), (0, 180, 0), -1)
            cv2.putText(overlay, action_text, (25, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            action_text_expiry -= 1

        cv2.imshow("DINOv3 Gesture Controller (Runtime)", overlay)

        # Check key presses (q to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Runtime controller closed.")

if __name__ == "__main__":
    main()
