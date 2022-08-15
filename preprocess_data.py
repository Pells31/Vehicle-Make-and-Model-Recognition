import pandas as pd
import numpy as np
import torch

from PIL import Image
import os
import argparse
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset


class CarsDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.labels = df.loc[:, "Classencoded"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if type(idx) is np.ndarray:
            idx = idx.item()

        if self.df.iloc[idx, 2] == "StanfordCars":
            filepath = self.root_dir + "StanfordCars/car_ims"
        else:
            filepath = self.root_dir + "VMMRdb/" + self.df.iloc[idx, 3]

        filepath = os.path.join(filepath, self.df.iloc[idx, 1])
        image = Image.open(filepath)

        # If greyscale, convert to RGB
        if image.mode == "L":
            image = image.convert(mode="RGB")

        label = self.df.loc[self.df.index[idx], "Classencoded"]
        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)

        return image, label


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        im, labels = self.dataset[self.indices[idx]]
        return self.transform(im), labels

    def __len__(self):
        return len(self.indices)


def create_stanford_df():
    cars_annos = loadmat("Data/StanfordCars/cars_annos.mat")

    annos = []
    class_names = []

    print("Iterating over Stanford annotations")
    for anno in cars_annos["annotations"]:
        for field in tqdm(anno):
            file = field[0].item(0).replace("car_ims/", "")
            annos.append([file, field[5].item(0)])

    print("Iterating over Stanford classnames")
    for class_name in cars_annos["class_names"]:
        for field in tqdm(class_name):
            class_names.append(field[0])

    df_classnames = pd.DataFrame(class_names, columns=["Classname"])
    df_annos = pd.DataFrame(annos, columns=["Filename", "ClassID"])
    df_annos["ClassID"] = df_annos["ClassID"] - 1

    df_merged = pd.merge(df_classnames, df_annos, left_index=True, right_on="ClassID")
    df_merged = df_merged.drop(["ClassID"], axis=1)
    df_merged["Datasetname"] = "StanfordCars"

    return df_merged


def create_vmmrdb_df(min_examples=30):
    data_list = []

    print("Iterating over VMMRdb dirs/files")
    for root, dirs, files in tqdm(os.walk("Data/VMMRdb", topdown=False)):
        classfolder = root.replace("Data/VMMRdb/", "")
        classname = root.replace("Data/VMMRdb/", "")
        classname = classname.replace("_", " ")
        classname = classname.title()
        # Only include classes with sufficient examples
        if len(files) >= min_examples:
            for file in tqdm(files):
                data_list.append([classname, file, classfolder])

    df = pd.DataFrame(data_list, columns=["Classname", "Filename", "Classfolder"])
    df["Datasetname"] = "VMMRdb"
    df = df[:-1]

    return df


def create_unified_df(df_stanford, df_vmmrdb):
    df = pd.concat([df_stanford, df_vmmrdb], ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    num_classes = df["Classname"].nunique()

    return df, num_classes


def create_dataloaders(df, batch_size=32):
    # Encode labels (for compatibility with Torch)
    df["Classencoded"] = df["Classname"].factorize()[0]

    df_class_count = df.groupby(by=["Classencoded"]).count()
    df_class_count = df.sort_values(by=["Classname"])

    # print(f"df length: {len(df)}")
    # ros = RandomOverSampler()
    # df_resampled, _ = ros.fit_resample(df, df["Classencoded"])
    # print(f"df_resampled length: {len(df_resampled)}")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomRotation(15),
                                           transforms.ToTensor(),
                                           transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                                           transforms.ColorJitter(brightness=.5, hue=.3),
                                           normalize])

    valid_test_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                                transforms.ToTensor(),
                                                normalize])

    full_dataset = CarsDataset(df=df,
                               root_dir="Data/")

    # Generate indices instead of using actual data
    train_val_indcs, test_indcs, _, _ = train_test_split(range(len(full_dataset)),
                                                         full_dataset.labels,
                                                         test_size=0.1,
                                                         stratify=full_dataset.labels)

    # generate subset based on indices
    train_val_split = Subset(full_dataset, train_val_indcs, transform=None)  # 0.9

    train_val_split_labels = []
    for idx in train_val_indcs:
        train_val_split_labels.append(full_dataset.labels[idx])

    train_indcs, val_indcs, train_labels, val_labels = train_test_split(range(len(train_val_split)),
                                                                        train_val_split_labels,
                                                                        test_size=0.111,
                                                                        stratify=train_val_split_labels)

    # Oversampling training set
    ros = RandomOverSampler()
    train_resampled_indcs, _ = ros.fit_resample(np.array(train_indcs).reshape(-1, 1), train_labels)

    train_split = Subset(full_dataset, train_resampled_indcs, train_transforms)  # 0.8
    val_split = Subset(full_dataset, val_indcs, valid_test_transforms)  # 0.1
    test_split = Subset(full_dataset, test_indcs, valid_test_transforms)  # 0.1

    # TODO: Code to check the above sequence is creating the proper splits/class distribution

    dataloaders = {
        "train": DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        "val": DataLoader(val_split, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        "test": DataLoader(test_split, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    }

    df.to_pickle("Data/dataloaders.pkl")

    return dataloaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batchsize", type=int, default=64, help="batch size")
    args = vars(parser.parse_args())

    df_stanford = create_stanford_df()
    df_vmmrdb = create_vmmrdb_df(min_examples=100)
    df, num_classes = create_unified_df(df_stanford, df_vmmrdb)

    print(f'num_classes: {num_classes}')

    batch_size = args["batchsize"]

    dataloaders = create_dataloaders(df, batch_size)


if __name__ == '__main__':
    main()
