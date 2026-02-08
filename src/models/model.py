import segmentation_models_pytorch as smp
import torch.nn as nn

def convert_bn_to_gn(module, num_groups=32):
    """
    Recursively replaces BatchNorm2d with GroupNorm.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            # replace with GroupNorm - use child.num_features to match the existing channel count
            num_channels = child.num_features
            groups = num_groups if num_channels % num_groups == 0 else num_channels # check for divisibility - specifically for decoder and segmentation head

            setattr(module, name, nn.GroupNorm(groups, child.num_features))
        else:
            convert_bn_to_gn(child, num_groups)

def get_model(num_classes=19, backbone="resnet50"):
    model = smp.Unet(
        encoder_name=backbone,
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )

    convert_bn_to_gn(model, num_groups=32)
    return model