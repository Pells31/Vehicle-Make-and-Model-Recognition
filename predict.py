import argparse
import torch
from torchvision import transforms
import pandas as pd

from skimage import io

from train import initialize_model
from preprocess_data import resize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="path to image")
    parser.add_argument("-m", "--model", type=str, help="model name")
    args = vars(parser.parse_args())

    path = args["path"]
    model_name = args["model"]

    img_transforms = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    img = io.imread(path)
    resize(path, size=224)
    img = img_transforms(img)
    img = img.to(device)
    img = img.unsqueeze(0)  # Add batch dimension (because single image)
    print(img.size())

    df = pd.read_pickle("Data/preprocessed_data.pkl")
    num_classes = df["Classname"].nunique()

    model, _ = initialize_model(model_name, num_classes, feature_extract=True)
    model.load_state_dict(torch.load("models/" + str(model_name), map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)

    class_encoded_matches = df.loc[df["Classencoded"] == pred.item()]
    class_name = class_encoded_matches.iloc[0, 0]

    print(class_name)


if __name__ == "__main__":
    main()
