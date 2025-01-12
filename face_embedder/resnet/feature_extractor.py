import torch
import torchvision.models as models

resnet50 = models.resnet50(pretrained=True)

feature_extractor = torch.nn.Sequential(
    *list(resnet50.children())[:-1],
    torch.nn.Flatten()
)

for param in feature_extractor.parameters():
    param.requires_grad = False
