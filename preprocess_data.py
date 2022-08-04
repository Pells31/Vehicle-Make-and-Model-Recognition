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


def main():
    # dataset = "StanfordCars"
    # preprocess_images(dataset)
    # dataset = "VMMRdb"
    # preprocess_images(dataset)

    cars_meta = loadmat("Data/StanfordCars/car_devkit/devkit/cars_meta.mat")
    cars_train_annos = loadmat("Data/StanfordCars/car_devkit/devkit/cars_train_annos.mat")
    cars_test_annos = loadmat("Data/StanfordCars/car_devkit/devkit/cars_test_annos_withlabels.mat")
    print(cars_meta)

    for anno in cars_train_annos['annotations']:
        for field in anno:
            print(str(field[4]) + " " + str(field[5]))

    for anno in cars_test_annos['annotations']:
        for field in anno:
            print(str(field[4]) + " " + str(field[5]))


if __name__ == "__main__":
    main()



