import argparse
import time

import pandas as pd
import torch
from torch import optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from torchvision.io import read_image, write_jpeg

from skimage import io
from sklearn.model_selection import train_test_split
from preprocess_data import create_unified_df, create_stanford_df, create_vmmrdb_df, preprocess_images

import os
import copy

from tqdm import tqdm

INPUT_SIZE = 224
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CarsDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.labels = df.iloc[:, 4]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.df.iloc[idx, 2] == "StanfordCars":
            filepath = self.root_dir + "StanfordCars/car_ims"
        else:
            filepath = self.root_dir + "VMMRdb/" + self.df.iloc[idx, 3]

        filepath = os.path.join(filepath,
                                self.df.iloc[idx, 1])
        image = io.imread(filepath)
        label = self.df.iloc[idx, 4]
        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)

        return image, label


def initialize_model(model_name, num_classes, feature_extract):
    if model_name == "resnet":
        weights = models.ResNet50_Weights.DEFAULT  # Init with the best available weights
        model = models.resnet50(weights=weights)

        set_parameter_requires_grad(model, feature_extract)

        fc_in_feats = model.fc.in_features
        model.fc = nn.Linear(fc_in_feats, num_classes)  # requires_grad=True by default

        model = model.to(device)

        return model, weights


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def train_model(model, data_loaders, criterion, optimizer, scheduler, num_epochs=100):
    since = time.time()
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        # print(f'Epoch {epoch}/{num_epochs - 1}')
        # print('-' * 15)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(data_loaders[phase].dataset)

            print("\n{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # Deep copy if best model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

    # Timer
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model


def test_accuracy(data_loaders, model):
    num_correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in data_loaders["test"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            preds = model(inputs)
            num_correct += (preds == labels).sum()
            total += labels.size(0)

    print(f"Test Accuracy of the model: {float(num_correct) / float(total) * 100:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("-b", "--batchsize", type=int, default=64, help="batch size")
    args = vars(parser.parse_args())

    df = pd.read_pickle("Data/preprocessed_data.pkl")
    num_classes = df["Classname"].nunique()

    image_transforms = transforms.Compose([transforms.ToTensor(),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    full_dataset = CarsDataset(df=df,
                               root_dir="Data/Processed/",
                               transform=image_transforms)

    # Generate indices instead of using actual data
    train_val_indcs, test_indcs, _, _ = train_test_split(range(len(full_dataset)),
                                                         full_dataset.labels,
                                                         test_size=0.1,
                                                         stratify=full_dataset.labels)

    # generate subset based on indices
    train_val_split = Subset(full_dataset, train_val_indcs)  # 0.9

    train_val_split_labels = [label for _, label in train_val_split]

    train_indcs, val_indcs, _, _ = train_test_split(range(len(train_val_split)),
                                                    train_val_split_labels,
                                                    test_size=0.111,
                                                    stratify=train_val_split_labels)

    train_split = Subset(full_dataset, train_indcs)  # 0.8
    val_split = Subset(full_dataset, val_indcs)  # 0.1
    test_split = Subset(full_dataset, test_indcs)  # 0.1

    # TODO: Code to check the above sequence is creating the proper splits/class distribution

    batch_size = args["batchsize"]
    dataloaders = {"train": DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=4),
                   "val": DataLoader(val_split, batch_size=batch_size, shuffle=True, num_workers=4),
                   "test": DataLoader(test_split, batch_size=batch_size, shuffle=True, num_workers=4)}

    model_name = "resnet"
    model, weights = initialize_model(model_name, num_classes, feature_extract=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    num_epochs = args["epochs"]
    best_model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=num_epochs)
    torch.save(best_model.state_dict(), "models/resnet50_" + str(num_epochs) + "epochs" + ".pt")

    test_accuracy(dataloaders, model)


if __name__ == "__main__":
    main()
