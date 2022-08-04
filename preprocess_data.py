import pandas as pd
import numpy as np
import os
from PIL import Image
from scipy.io import loadmat

SIZE = 224


def resize(filepath, size=SIZE):
    img = Image.open(filepath)

    # Maintains Aspect Ratio
    img.thumbnail((SIZE, SIZE), Image.ANTIALIAS)
    new_width, new_height = img.size

    # Add padding
    result = Image.new(img.mode, (SIZE, SIZE), 0)
    if new_height < SIZE:
        result.paste(img, (0, SIZE - new_height))

    img.close()
    return result


def preprocess_images(dir):
    count = 0
    orig_dir_path = "Data/" + dir
    for root, dirs, files in os.walk(orig_dir_path, topdown=False):
        for file in files:

            # Only process pictures
            if os.path.splitext(file)[-1].lower() == ".jpg":
                orig_pic_path = root + "/" + file
                result = resize(orig_pic_path)

                # Create directory structure if doesn't exist
                if not os.path.isdir("Data/Processed/" + root[5:]):
                    os.makedirs("Data/Processed/" + root[5:])

                # Save image if not all black
                if result.getbbox():
                    result.save("Data/Processed/" + root[5:] + "/" + file)

                result.close()
                count += 1
                if count % 10000 == 0:
                    print("Preprocessed " + str(count) + " pictures")


def create_stanford_df():
    cars_meta = loadmat("Data/StanfordCars/car_devkit/devkit/cars_meta.mat")
    cars_annos = loadmat("Data/StanfordCars/cars_annos.mat")

    annos = []
    class_names = []

    for anno in cars_annos["annotations"]:
        for field in anno:
            annos.append([field[0].item(0).replace("car_ims/", ""), field[5].item(0)])

    for class_name in cars_annos["class_names"]:
        for field in class_name:
            class_names.append(field[0])

    df_classnames = pd.DataFrame(class_names, columns=["Classname"])
    df_annos = pd.DataFrame(annos, columns=['Filename', "ClassID"])
    df_annos["ClassID"] = df_annos["ClassID"] - 1

    df_merged = pd.merge(df_classnames, df_annos, left_index=True, right_on="ClassID")
    df_merged = df_merged.drop(["ClassID"], axis=1)
    return df_merged


def create_vmmrdb_df():
    data_list = []
    for root, dirs, files in os.walk("Data/Processed/VMMRdb", topdown=False):
        classname = root.replace("Data/Processed/VMMRdb/", "")
        classname = classname.replace("_", " ")
        classname = classname.title()
        for file in files:
            data_list.append([classname, file])

    df = pd.DataFrame(data_list, columns=["Classname", "Filename"])
    df = df[:-1]
    return df


def create_unified_df():
    df_stanford = create_stanford_df()
    df_vmmrdb = create_vmmrdb_df()
    df = pd.concat([df_stanford, df_vmmrdb], ignore_index=True)
    return df


def main():
    dataset = "StanfordCars"
    # preprocess_images(dataset)
    # dataset = "VMMRdb"
    # preprocess_images(dataset)


if __name__ == "__main__":
    main()



