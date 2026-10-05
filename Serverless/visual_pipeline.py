import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path

script_dir = Path(__file__).parent
image_path = script_dir / "pants.jpg" 

# Load the local image safely
original_image = Image.open(image_path)

# --- STEP 1: The Original Raw Image ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title("1. Original Raw Image")
plt.axis('off')

# --- STEP 2: Resized to 150x150 ---
resized_image = original_image.resize((150, 150))

plt.subplot(1, 3, 2)
plt.imshow(resized_image)
plt.title("2. Resized Thumbnail\n(Exact 150x150 pixels)")
plt.axis('off')

# --- STEP 3: Preprocessed Tensor Visualized ---
# Emulate Xception's preprocessing: scale pixels from [0, 255] to [-1, 1]
img_array = np.array(resized_image, dtype=np.float32)
tensor_visual = (img_array / 127.5) - 1.0

# Shift back from [-1, 1] to [0, 1] just so Matplotlib can display it correctly
display_version = (tensor_visual + 1.0) / 2.0

plt.subplot(1, 3, 3)
plt.imshow(display_version)
plt.title("3. Preprocessed Tensor\n(Fed into Model Input Slot)")
plt.axis('off')

plt.tight_layout()
plt.show()