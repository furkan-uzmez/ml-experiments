import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_model(num_classes=2, pretrained=True):
    if pretrained:
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    else:
        model = resnet50(weights=None)
        
    # Replace the FC layer for binary/multi-class classification
    in_features = model.fc.in_features
    # We use num_classes=2 as a default
    model.fc = nn.Linear(in_features, num_classes)
    return model
