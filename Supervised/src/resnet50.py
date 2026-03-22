import torch.utils.checkpoint
import torch
import torch.nn as nn
import torchvision.models as models

class resnet50(nn.Module):
    def __init__(self, mode, use_checkpoint, pretrained=False):
        super(resnet50, self).__init__()

        self.mode = mode
        self.use_checkpoint = use_checkpoint

        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.resnet50(weights=None)

        self.encoder_out_features = self.backbone.fc.in_features
        
    def load_weights(self, path, device):
        checkpoint = torch.load(path, map_location=device)

        state_dict = checkpoint.get("state_dict", checkpoint)

        errors = []

        # Classifier head 2 classes
        self.fit_classifier_head(2)
        try:
            self.load_state_dict(state_dict)
            return
        except Exception as e:
            errors.append(("classifier_head_2", str(e)))

        # Classifier head 1000 classes
        self.fit_classifier_head(1000)
        try:
            self.load_state_dict(state_dict)
            return
        except Exception as e:
            errors.append(("classifier_head_1000", str(e)))

        raise ValueError(
            f"Failed to load weights from {path}. "
            f"Tried heads: {errors}"
    )

    def freeze_encoder(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def fit_classifier_head(self, num_classes):
        self.backbone.fc = nn.Linear(self.encoder_out_features, num_classes, bias=True)

    def _forward_impl(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)

        return x
    
    def _forward_impl_checkpoint(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = torch.utils.checkpoint.checkpoint(self.backbone.layer1, x, use_reentrant=False)
        x = torch.utils.checkpoint.checkpoint(self.backbone.layer2, x, use_reentrant=False)
        x = torch.utils.checkpoint.checkpoint(self.backbone.layer3, x, use_reentrant=False)
        x = torch.utils.checkpoint.checkpoint(self.backbone.layer4, x, use_reentrant=False)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)

        return x

    def forward(self, x):
        if self.use_checkpoint:
            return self._forward_impl_checkpoint(x)
        return self._forward_impl(x)
