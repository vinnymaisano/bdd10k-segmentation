import segmentation_models_pytorch as smp
import torch.nn as nn

def get_model(num_classes=19, backbone="resnet50", use_gn=False):
    model = smp.Unet(
        encoder_name=backbone,
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )
    
    # convert batch norm layers to group norm
    if use_gn:
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # Split the path to the layer
                parts = name.split(".")
                parent_name = ".".join(parts[:-1])
                layer_name = parts[-1]
                
                # Access the parent module
                parent = dict(model.named_modules())[parent_name]
                
                # Determine number of groups (must divide num_channels)
                num_channels = module.num_features
                # 32 is standard, but some early layers might have fewer channels
                groups = 32 if num_channels % 32 == 0 else 16
                if num_channels < groups: groups = num_channels
                
                # Replace BN with GN
                setattr(parent, layer_name, nn.GroupNorm(groups, num_channels))
                
    return model