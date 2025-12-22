import torch
from torchvision.io import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

img = read_image("assets/street.jpg").float() / 255.0  # [C,H,W] in 0..1
with torch.no_grad():
    preds = model([img])[0]  # dict with boxes, labels, scores

# Example: keep confident detections
keep = preds["scores"] > 0.7
boxes = preds["boxes"][keep].tolist()
labels = preds["labels"][keep].tolist()
scores = preds["scores"][keep].tolist()
print(len(boxes), "detections above threshold")