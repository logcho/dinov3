import sys
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import urllib.request
from tqdm import tqdm

try:
    import cv2
except ImportError:
    print("Error: 'opencv-python' (cv2) is not installed.")
    sys.exit(1)

# Path to the local DINOv3 repository directory
REPO_DIR = "/Users/loganchoi/Desktop/dinov3/dinov3"
sys.path.append(REPO_DIR)

def create_synthetic_video(filename="sample.mp4", num_frames=120, size=448):
    print("Generating synthetic video locally...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (size, size))
    
    # Bouncing ball parameters
    x, y = size // 2, size // 2
    vx, vy = 8, 11
    radius = 35
    
    for _ in range(num_frames):
        # Create a dark background frame
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame[:, :] = [40, 40, 40] # Gray background
        
        # Update ball position and bounce off walls
        x += vx
        y += vy
        if x - radius < 0 or x + radius > size:
            vx = -vx
            x = np.clip(x, radius, size - radius)
        if y - radius < 0 or y + radius > size:
            vy = -vy
            y = np.clip(y, radius, size - radius)
            
        # Draw stationary obstacles (green, red) and the moving ball (blue)
        cv2.circle(frame, (size // 4, size // 4), 30, (0, 200, 0), -1)  # Green obstacle
        cv2.circle(frame, (3 * size // 4, 3 * size // 4), 50, (0, 0, 200), -1)  # Red obstacle
        cv2.circle(frame, (int(x), int(y)), radius, (200, 0, 0), -1)  # Blue bouncing ball
        
        out.write(frame)
        
    out.release()
    print(f"Synthetic video saved to '{filename}'")

def main():
    # 1. Device configuration (use MPS on Apple Silicon if available, otherwise CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU acceleration (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 2. Load DINOv3 backbone
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

    # 3. Ensure we have a sample video
    input_video_path = "sample.mp4"
    output_video_path = "processed_output.mp4"
    create_synthetic_video(input_video_path)

    # 4. Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_video_path}'")
        return

    # Read video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video loaded: {total_frames} frames at {fps} FPS.")

    # 5. Define output size and setup VideoWriter
    display_size = 448 # Size of each square frame
    # Output video will be side-by-side (width = 2 * display_size, height = display_size)
    out_width = display_size * 2
    out_height = display_size

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))
    print(f"Output video writer configured. Saving to '{output_video_path}'...")

    # 6. Transform for DINOv3
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 7. Process frames loop
    try:
        for _ in tqdm(range(total_frames), desc="Processing Video Frames"):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Crop frame to square center
            h, w, _ = frame.shape
            crop_size = min(h, w)
            start_x = (w - crop_size) // 2
            start_y = (h - crop_size) // 2
            cropped_frame = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]

            # Convert to PIL and preprocess
            rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            # Get features
            with torch.no_grad():
                features = model.forward_features(input_tensor)
                patch_tokens = features["x_norm_patchtokens"].squeeze(0)

            # Calculate dynamic grid dimensions
            grid_h = input_tensor.shape[2] // model.patch_size
            grid_w = input_tensor.shape[3] // model.patch_size

            # Perform PCA via low-rank SVD in PyTorch
            mean = patch_tokens.mean(dim=0, keepdim=True)
            centered = patch_tokens - mean
            U, S, V = torch.pca_lowrank(centered, q=3, center=False)
            projected = torch.matmul(centered, V)

            # Normalize PCA components
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
            pca_grid = pca_features.reshape(grid_h, grid_w, 3)

            # Resize PCA image back to the display size
            pca_img = Image.fromarray(pca_grid).resize((display_size, display_size), Image.NEAREST)
            pca_frame = np.array(pca_img)
            pca_frame_bgr = cv2.cvtColor(pca_frame, cv2.COLOR_RGB2BGR)

            # Resize original cropped square frame to match display size
            orig_resized = cv2.resize(cropped_frame, (display_size, display_size))

            # Stack side-by-side
            combined_view = np.hstack((orig_resized, pca_frame_bgr))

            # Write combined frame to output file
            out.write(combined_view)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")

    finally:
        # Clean up files
        cap.release()
        out.release()
        print(f"\nSuccessfully finished! Processed video saved to '{output_video_path}'.")

if __name__ == "__main__":
    main()
