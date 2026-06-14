import sys
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Make sure we can import opencv-python
try:
    import cv2
except ImportError:
    print("Error: 'opencv-python' (cv2) is not installed in this environment.")
    print("Please install it by running: pip install opencv-python")
    sys.exit(1)

# 1. Path to the local DINOv3 repository directory
REPO_DIR = "/Users/loganchoi/Desktop/dinov3/dinov3"
sys.path.append(REPO_DIR)

def main():
    # 2. Select device (use MPS on Apple Silicon if available, otherwise CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU acceleration (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 3. Load the ViT-Small model (lightweight and fast for real-time webcam)
    print("Loading DINOv3 ViT-S/16 model weights from Hugging Face...")
    model = torch.hub.load(
        REPO_DIR, 
        'dinov3_vits16', 
        source='local', 
        weights='https://huggingface.co/jaychempan/dinov3/resolve/main/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
    )
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # 4. Define ImageNet preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((560, 560)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


    # 5. Initialize camera (try multiple indices in case of Continuity Camera or external webcams)
    cap = None
    for camera_idx in [0, 1, 2]:
        print(f"Trying to open camera index {camera_idx}...")
        temp_cap = cv2.VideoCapture(camera_idx)
        if temp_cap.isOpened():
            # Try to grab a few frames to make sure it's active and has permission
            success = False
            for _ in range(10):
                ret, frame = temp_cap.read()
                if ret and frame is not None:
                    success = True
                    break
                cv2.waitKey(50)
            if success:
                cap = temp_cap
                print(f"Successfully connected to camera index {camera_idx}!")
                break
            else:
                temp_cap.release()

    if cap is None:
        print("\n" + "="*60)
        print("Error: Could not read a valid frame from any camera device.")
        print("\nOn macOS, this is usually caused by one of the following:")
        print("1. Camera Permissions: Your Terminal or VS Code lacks camera access.")
        print("   Go to System Settings -> Privacy & Security -> Camera and enable access for your terminal/editor.")
        print("2. Busy Camera: Another app (Zoom, Teams, FaceTime, Web Browser) is currently using the camera.")
        print("3. Continuity Camera: Try turning off Apple Continuity Camera in your phone/Mac settings.")
        print("="*60 + "\n")
        return

    cv2.namedWindow("DINOv3 Live PCA Semantic Segmentation", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to read frame from webcam.")
            break

        # Flip frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Get square crop from center of frame
        h, w, _ = frame.shape
        crop_size = min(h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        cropped_frame = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]

        # Convert BGR (OpenCV) to RGB (PIL) for the model
        rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # Preprocess frame tensor and move to device
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        # 6. Run model forward pass to get patch tokens
        with torch.no_grad():
            features = model.forward_features(input_tensor)
            # patch_tokens shape: [num_patches, embed_dim]
            patch_tokens = features["x_norm_patchtokens"].squeeze(0)

        # Calculate grid dimensions dynamically based on input resolution and patch size (16)
        grid_h = input_tensor.shape[2] // model.patch_size
        grid_w = input_tensor.shape[3] // model.patch_size

        # 7. Perform SVD-based PCA in PyTorch directly for speed
        # Center the features
        mean = patch_tokens.mean(dim=0, keepdim=True)
        centered = patch_tokens - mean
        
        # Calculate low-rank PCA projection using PyTorch's native SVD solver
        U, S, V = torch.pca_lowrank(centered, q=3, center=False)
        projected = torch.matmul(centered, V) # shape: [num_patches, 3]

        # 8. Normalize components to [0, 255] and construct RGB semantic image
        projected_np = projected.cpu().numpy()
        pca_features = np.zeros_like(projected_np)
        for i in range(3):
            c_min = projected_np[:, i].min()
            c_max = projected_np[:, i].max()
            if c_max - c_min > 1e-5:
                pca_features[:, i] = 255.0 * (projected_np[:, i] - c_min) / (c_max - c_min)
            else:
                pca_features[:, i] = 0.0

        pca_features = pca_features.astype(np.uint8)

        # Reshape to dynamic grid dimensions with 3 channels (RGB)
        pca_grid = pca_features.reshape(grid_h, grid_w, 3)

        # 9. Resize the segmentation map to match the webcam view size
        display_size = 448
        pca_img = Image.fromarray(pca_grid).resize((display_size, display_size), Image.NEAREST)
        pca_frame = np.array(pca_img)

        # Convert PCA image from RGB to BGR for OpenCV
        pca_frame_bgr = cv2.cvtColor(pca_frame, cv2.COLOR_RGB2BGR)

        # Resize original cropped square frame to match the display size
        orig_resized = cv2.resize(cropped_frame, (display_size, display_size))

        # Concatenate original frame and PCA segmentation map side-by-side
        combined_view = np.hstack((orig_resized, pca_frame_bgr))

        # Render combined frame
        cv2.imshow("DINOv3 Live PCA Semantic Segmentation", combined_view)

        # Handle keyboard exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam pipeline closed.")

if __name__ == "__main__":
    main()
