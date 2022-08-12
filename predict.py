import argparse
import torch
from torchvision import transforms
import pandas as pd

from skimage import io

from train import initialize_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(file, model_name="resnet50", k=5):

    img_transforms = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    img = io.imread(file)
    img = img_transforms(img)
    img = img.to(device)
    img = img.unsqueeze(0)  # Add batch dimension (because single image)
    print(img.size())

    df = pd.read_pickle("Data/preprocessed_data.pkl")
    num_classes = df["Classname"].nunique()

    model, _ = initialize_model(model_name[:8], num_classes, feature_extract=True)
    model.load_state_dict(torch.load("models/" + str(model_name), map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(img)
        _, preds = torch.topk(output, k)

    preds = torch.transpose(preds, 0, 1)
    preds = preds.cpu()  # Send tensor to cpu
    preds = pd.DataFrame(preds.numpy(), columns=["Classencoded"])  # Convert to dataframe

    class_encoded_matches = pd.merge(df, preds, how="inner")
    classname_matches = class_encoded_matches["Classname"].unique()

    return classname_matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="path to image")
    parser.add_argument("-m", "--model", type=str, help="model name")
    parser.add_argument("-k", "--topk", type=int, help="top k predictions")
    args = vars(parser.parse_args())

    path = args["path"]
    model_name = args["model"]
    k = args["topk"]

    classname_matches = predict(path, model_name, k)
    print(classname_matches)


if __name__ == "__main__":
    main()
