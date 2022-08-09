import pandas as pd
import numpy as np
import os
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm


def resize(filepath, size=224):
    img = Image.open(filepath)

    # If greyscale, convert to RGB
    if img.mode == "L":
        img = img.convert(mode="RGB")

    # Maintains Aspect Ratio
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    new_width, new_height = img.size

    # Add padding
    result = Image.new(img.mode, (size, size), 0)
    if new_height < size:
        result.paste(img, (0, size - new_height))

    img.close()
    return result


def preprocess_images(dataset_dir):
    orig_dir_path = "Data/" + dataset_dir

    not_saved = []

    if dataset_dir == "VMMRdb":
        for root, dirs, files in tqdm(os.walk(orig_dir_path, topdown=False)):
            for file in files:
                # Only process pictures
                if os.path.splitext(file)[-1].lower() == ".jpg":
                    orig_pic_path = root + "/" + file
                    result = resize(orig_pic_path)

                    # Create directory structure if doesn't exist
                    if not os.path.isdir("Data/Processed/" + root[5:]):
                        os.makedirs("Data/Processed/" + root[5:])

                    # Save image if not all black, else record to not saved list
                    filepath = "Data/Processed/" + root[5:] + "/" + file
                    if result.getbbox():
                        result.save(filepath)
                    else:
                        not_saved.append(file)
                    result.close()

    else:
        for root, dirs, files in os.walk(orig_dir_path, topdown=False):
            for file in tqdm(files):
                # Only process pictures
                if os.path.splitext(file)[-1].lower() == ".jpg":
                    orig_pic_path = root + "/" + file
                    result = resize(orig_pic_path)

                    # Create directory structure if doesn't exist
                    if not os.path.isdir("Data/Processed/" + root[5:]):
                        os.makedirs("Data/Processed/" + root[5:])

                    # Save image if not all black, else record to not saved list
                    filepath = "Data/Processed/" + root[5:] + "/" + file
                    if result.getbbox():
                        result.save(filepath)
                    else:
                        not_saved.append(file)
                    result.close()

    return not_saved


def create_stanford_df(not_saved_list):
    cars_annos = loadmat("Data/StanfordCars/cars_annos.mat")

    annos = []
    class_names = []
    for anno in cars_annos["annotations"]:
        for field in anno:
            file = field[0].item(0).replace("car_ims/", "")
            annos.append([file, field[5].item(0)])

    for class_name in cars_annos["class_names"]:
        for field in class_name:
            class_names.append(field[0])

    df_classnames = pd.DataFrame(class_names, columns=["Classname"])
    df_annos = pd.DataFrame(annos, columns=["Filename", "ClassID"])
    df_annos["ClassID"] = df_annos["ClassID"] - 1

    df_merged = pd.merge(df_classnames, df_annos, left_index=True, right_on="ClassID")
    df_merged = df_merged.drop(["ClassID"], axis=1)
    df_merged["Datasetname"] = "StanfordCars"

    print("Stanford df BEFORE removing not_saved:")
    print(df_merged.shape[0])

    # Remove not_saved images
    for not_saved in not_saved_list:
        df_merged = df_merged[df_merged["Filename"] != not_saved]

    print("Stanford df AFTER removing not_saved:")
    print(df_merged.shape[0])

    return df_merged


def create_vmmrdb_df(not_saved_list, min_examples=10):
    data_list = []
    for root, dirs, files in os.walk("Data/Processed/VMMRdb", topdown=False):
        classfolder = root.replace("Data/Processed/VMMRdb/", "")
        classname = root.replace("Data/Processed/VMMRdb/", "")
        classname = classname.replace("_", " ")
        classname = classname.title()
        # Only include classes with sufficient examples
        if len(files) >= min_examples:
            for file in files:
                data_list.append([classname, file, classfolder])

    df = pd.DataFrame(data_list, columns=["Classname", "Filename", "Classfolder"])
    df["Datasetname"] = "VMMRdb"
    df = df[:-1]

    print("VMMRdb df BEFORE removing not_saved:")
    print(df.shape[0])

    # Remove not_saved images
    for not_saved in not_saved_list:
        df = df[df["Filename"] != not_saved]

    print("VMMRdb df AFTER removing not_saved:")
    print(df.shape[0])

    return df


def create_unified_df(df_stanford, df_vmmrdb):
    df = pd.concat([df_stanford, df_vmmrdb], ignore_index=True)

    # Encode labels (for compatibility with Torch)
    df["Classencoded"] = df["Classname"].factorize()[0]

    return df


def main():
    dataset = "StanfordCars"
    not_saved_stanford = preprocess_images(dataset)
    dataset = "VMMRdb"
    not_saved_vmmr = preprocess_images(dataset)

    df_stanford = create_stanford_df(not_saved_stanford)
    df_vmmrdb = create_vmmrdb_df(not_saved_vmmr, min_examples=10)

    df = create_unified_df(df_stanford, df_vmmrdb)
    df.to_pickle("Data/preprocessed_data.pkl")


if __name__ == '__main__':
    main()
