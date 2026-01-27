import segmentation_models_pytorch as smp
import torch

def get_model(num_classes=19, backbone="resnet50"):
    # create u-net model
    model = smp.Unet(
        encoder_name=backbone,
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )
            
    return model