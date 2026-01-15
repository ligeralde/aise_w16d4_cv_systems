import torch
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.transforms.functional import to_tensor
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------
# Load the image via PIL (macOS-safe)
# -------------------------
img_path = "assets/street.jpg"
img = Image.open(img_path).convert("RGB")
img_tensor = to_tensor(img)  # [C,H,W] in [0,1]

# -------------------------
# Load pretrained DeepLabV3 model
# -------------------------
weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet50(weights=weights)
model.eval()

# -------------------------
# Run inference
# -------------------------
with torch.no_grad():
    out = model(img_tensor.unsqueeze(0))["out"]  # [1, C, H, W]

# -------------------------
# Get predicted segmentation mask
# -------------------------
mask = out.argmax(1)[0]  # [H,W], class IDs per pixel
print("Mask shape:", mask.shape)

# -------------------------
# Optional visualization
# -------------------------
# DeepLabV3 COCO classes (for 21 classes including background)
COCO_SEG_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "dining table", "dog", "horse", "motorbike",
    "person", "potted plant", "sheep", "sofa", "train",
    "tv/monitor"
]

# Map each class ID to a color for visualization
import numpy as np
colors = np.random.randint(0, 255, size=(len(COCO_SEG_CLASSES), 3), dtype=np.uint8)
mask_color = colors[mask.numpy()]

plt.figure(figsize=(10,10))
plt.imshow(mask_color)
plt.axis("off")
plt.show()
