import torch.nn as nn
import segmentation_models_pytorch as smp


class DeepLabV3PResNet34(nn.Module):
    def __init__(self, num_classes: int = 8):
        super(DeepLabV3PResNet34, self).__init__()
        self.layer = smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
        self.activation = nn.Sigmoid()
        self.n_classes = num_classes

    def forward(self, x):
        return self.layer(x)
