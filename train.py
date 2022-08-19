import argparse
import time

import torch
from torch import optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models

from tqdm import tqdm
from copy import deepcopy

from preprocess_data import create_stanford_df, create_vmmrdb_df, create_unified_df, create_dataloaders
from utils import plot_loss_curves
from preprocess_data import preprocess_images

cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_model(model_name, num_classes, feature_extract=True):
    if model_name == "resnet152":
        weights = models.ResNet152_Weights.DEFAULT  # Init with the best available weights
        model = models.resnet152(weights=weights)

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)

    elif model_name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT
        model = models.resnet34(weights=weights)

    else:
        raise Exception("Backbone model not recognized.")

    set_parameter_requires_grad(model, feature_extract)

    fc_in_feats = model.fc.in_features
    model.fc = nn.Linear(fc_in_feats, num_classes)  # requires_grad=True by default

    model = model.to(device)

    return model, weights


def set_parameter_requires_grad(model, feature_extracting=True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def train_model(model, model_name, data_loaders, criterion, optimizer, scheduler, num_epochs=100):
    since = time.time()
    best_acc = 0.0
    train_losses = []
    train_accs = []
    val_accs = []
    val_losses = []

    for epoch in tqdm(range(num_epochs)):

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

                running_loss += float(loss) * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects / len(data_loaders[phase].dataset)

            print("\n{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "train":
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)

            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)
                # Deep copy if best model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_weights = deepcopy(model.state_dict())
                    torch.save(model.state_dict(), "models/" + str(model_name) + "_" + str(num_epochs) + "epochs" + ".pt")
                    print("Model saved!")

    # Timer
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model, train_losses, train_accs, val_losses, val_accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("-b", "--batchsize", type=int, default=64, help="batch size")
    parser.add_argument("-m", "--model", type=str, help="model name")
    parser.add_argument("-lr", "--learning_rate", type=float, help="learning rate")
    args = vars(parser.parse_args())

    dataset = "StanfordCars"
    not_saved_stanford = preprocess_images(dataset)
    dataset = "VMMRdb"
    not_saved_vmmr = preprocess_images(dataset)

    df_stanford = create_stanford_df(not_saved_stanford)
    df_vmmrdb = create_vmmrdb_df(not_saved_vmmr, min_examples=100)
    df, num_classes = create_unified_df(df_stanford, df_vmmrdb)
    print(f'df created!')
    print(f'num_classes: {num_classes}')

    model_name = args["model"]
    model, weights = initialize_model(model_name, num_classes, feature_extract=False)
    print(f'model initialized!')

    lr = args["learning_rate"]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = args["epochs"]
    batch_size = args["batchsize"]
    dataloaders = create_dataloaders(df, batch_size)
    print(f'dataloaders created!')

    best_model, train_losses, train_accs, val_losses, val_accs = \
        train_model(model, model_name, dataloaders, criterion, optimizer, scheduler, num_epochs)

    plot_loss_curves(train_losses, train_accs, val_losses, val_accs, model_name)


if __name__ == "__main__":
    main()
