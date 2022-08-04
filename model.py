import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.io import read_image, write_jpeg

from skimage import io, transform
from preprocess_data import create_unified_df

import numpy as np
import copy
import os

INPUT_SIZE = 224
cudnn.benchmark = True


class CarsDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.df.iloc[idx, 1])
        image = io.imread(img_name)
        label = self.df.iloc[idx, 0]

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': label}


def initialize_model(model_name, num_classes, feature_extract):
    if model_name == "resnet":
        weights = models.ResNet50_Weights.DEFAULT  # Init with the best available weights
        model = models.resnet50(weights=weights)
        transforms = weights.transforms()  # Inference Transforms

        set_parameter_requires_grad(model, feature_extract)

        fc_in_feats = model.fc.in_features
        model.fc = nn.Linear(fc_in_feats, num_classes)

        return model, transforms, weights


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


def train_model(model, criterion, optimizer, scheduler, num_epochs=100):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()


def main():
    df = create_unified_df()
    print(df.head)
    num_classes = df["Classname"].nunique()

    model_name = "resnet"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Device: " + str(device) + "\n")
    model, transforms, weights = initialize_model(model_name, num_classes, feature_extract=True)
    # print("Transforms: " + str(transforms) + "\n")
    # print("Model: " + str(model) + "\n")

    model.eval()

    cars_dataset = CarsDataset(df=df,
                               root_dir="Data/Processed/StanfordCars/car_ims/",
                               transform=ToTensor())

    for i in range(1):
        sample = cars_dataset[i]
        print(i, sample["image"].size(), sample["label"])

    # batch = transforms(img).unsqueeze(dim=0)
    # print(transforms)
    #
    # print("Batch.size: " + str(batch.size()))
    # pred = model(batch).squeeze(0).softmax(0)
    # print("Pred.size: " + str(pred.size()))
    # class_id = pred.argmax().item()
    # print(class_id)
    # score = pred[class_id].item()
    # category_name = weights.meta["categories"][class_id]
    # print(f"{category_name}: {100 * score:.1f}%")


if __name__ == "__main__":
    main()




