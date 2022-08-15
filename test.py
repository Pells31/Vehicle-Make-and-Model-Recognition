import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Add F1, RUC


def test_accuracy(data_loaders, model):
    num_correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in data_loaders["test"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            print(preds.size())
            print(labels.size())

            num_correct += (preds == labels).sum()
            total += labels.size(0)

    print(f"Test Accuracy of the model: {float(num_correct) / float(total) * 100:.2f}")


def main():
    test_accuracy()


if __name__ == "__main__":
    main()
