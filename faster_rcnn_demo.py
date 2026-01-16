import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_tensor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -------------------------
# Load the image 
# -------------------------
img_path = "path_to_your_image"
img = Image.open(img_path).convert("RGB")
img_tensor = to_tensor(img)  # [C,H,W] in [0,1]

# -------------------------
# 2️Load pretrained Faster R-CNN model
# -------------------------
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()  # inference mode

# -------------------------
# Run inference
# -------------------------
with torch.no_grad():
    preds = model([img_tensor])[0]

# -------------------------
# Filter confident detections
# -------------------------
threshold = 0.7
keep = preds["scores"] > threshold
boxes = preds["boxes"][keep].tolist()
labels = preds["labels"][keep].tolist()
scores = preds["scores"][keep].tolist()

# -------------------------
# COCO class names
# -------------------------
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# -------------------------
# Print results
# -------------------------
print(f"{len(boxes)} detections above {threshold} confidence:")
for box, label, score in zip(boxes, labels, scores):
    print(f"Label: {COCO_INSTANCE_CATEGORY_NAMES[label]}, Score: {score:.2f}, Box: {box}")

# -------------------------
# Optional visualization
# -------------------------
fig, ax = plt.subplots(1, figsize=(12,8))
ax.imshow(img)

for box, label in zip(boxes, labels):
    x1, y1, x2, y2 = box
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                             linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(x1, y1-5, COCO_INSTANCE_CATEGORY_NAMES[label], color='yellow', fontsize=12, weight='bold')

plt.axis('off')
plt.show()
