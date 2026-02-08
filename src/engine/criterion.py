import torch
import torch.nn as nn
from src.engine.device import get_device

device = get_device()

# class weights
bdd_weights = torch.tensor([
    0.8,   # road
    1.2,   # sidewalk
    1.5,   # building
    2.0,   # wall
    2.0,   # fence
    2.0,   # pole
    2.5,   # traffic light
    2.5,   # traffic sign
    1.2,   # vegetation
    1.5,   # terrain
    1.0,   # sky
    2.0,   # person
    3.0,   # rider (very rare)
    1.2,   # car
    2.5,   # truck
    3.0,   # bus
    3.0,   # train
    3.0,   # motorcycle
    2.0    # bicycle
])

def get_criterion():
    """
    Standard Cross Entropy for multi-class segmentation.
    ignore_index=255 ensures "void" pixels won't ruin the training.
    """
    return nn.CrossEntropyLoss(weight=bdd_weights.to(device), ignore_index=255)

def get_optimizer(model, lr=1e-4, weight_decay=1e-5):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def get_class_weights():
    return bdd_weights