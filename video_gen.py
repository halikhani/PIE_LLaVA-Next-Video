from PIL import Image
import imageio
import os

# Directory containing image frames
image_dir = 'TMP_SCENE_DIR'

# Get list of image files in the directory
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')])

# Define the path and name for the output video
output_video_path = 'sample_scene.mp4'

# Define the frames per second (fps) for the video
fps = 24

# List to hold all frames
frames = []

for image_file in image_files:
    # Load image using PIL
    img = Image.open(os.path.join(image_dir, image_file))
    # Optionally, process the image (resize, add effects, etc.)
    # img = img.resize((width, height))
    # Convert image to numpy array and append to frames list
    frames.append(imageio.imread(os.path.join(image_dir, image_file)))

# Create a video from frames
imageio.mimsave(output_video_path, frames, fps=fps)

print(f"Video saved at {output_video_path}")
