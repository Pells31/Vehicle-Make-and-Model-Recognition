import pandas as pd
import os
from scipy.io import loadmat
from tqdm import tqdm


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


def main():

    df_stanford = create_stanford_df()
    df_vmmrdb = create_vmmrdb_df(min_examples=24)

    df = pd.concat([df_stanford, df_vmmrdb], ignore_index=True)

    # Encode labels (for compatibility with Torch)
    df["Classencoded"] = df["Classname"].factorize()[0]

    df.to_pickle("Data/preprocessed_data.pkl")


if __name__ == '__main__':
    main()
