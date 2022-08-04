import torch
import torch.nn as nn
from torchvision import models


def initialize_model(model_name, num_classes, feature_extract):
    if model_name == "resnet":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)

        set_parameter_requires_grad(model, feature_extract)

        fc_in_feats = model.fc.in_features
        print("fc_in_feats: " + str(fc_in_feats))
        model.fc = nn.Linear(fc_in_feats, num_classes)

        input_size = 224

        return model, input_size


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# class CNN(nn.module):
#     def __init__(self, fine_tune=False):
#         super(CNN, self).__init__()
#         weights = models.ResNet50_Weights.DEFAULT
#         resnet = models.resnet50(weights=weights)
#
#         # Fine-tune, or freeze parameters
#         if fine_tune:
#             for param in resnet.parameters():
#                 param.requires_grad = True
#         else:
#             for param in resnet.parameters():
#                 param.requires_grad = False


def main():
    model_name = "resnet"
    num_classes = 10000
    model, input_size = initialize_model(model_name, num_classes, feature_extract=True)
    print(model)


if __name__ == "__main__":
    main()




