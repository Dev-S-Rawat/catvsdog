import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights



def get_model(num_classes = 2):
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    # since mobileNet v3 has a input size of 1024 we only want to use 2 classes therefore we do-
    in_features = model.classifier[
        3].in_features  # how many signals are coming into mobileNet from previous layer ? « 1024
    model.classifier[3] = nn.Linear(in_features, 2)  # « output to only 2 nodes

    return model