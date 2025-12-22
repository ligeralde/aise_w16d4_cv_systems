import torch
from torchvision.io import read_image
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet50(weights=weights)
model.eval()

img = read_image("assets/street.jpg").float() / 255.0
# simple normalization via weights meta if you want to be more exact later
with torch.no_grad():
    out = model(img.unsqueeze(0))["out"]  # [1, C, H, W]
mask = out.argmax(1)[0]  # [H,W] class id per pixel
print(mask.shape)