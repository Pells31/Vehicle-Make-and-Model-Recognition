from matplotlib import pyplot as plt
import numpy as np


def plot_loss_curves(train_losses, train_accs, val_losses, val_accs, model_name):

    epochs = np.linspace(1, len(train_losses), num=len(train_losses))

    f, ax = plt.subplots(1, 2, figsize=(14, 10))
    ax[0].plot(epochs, train_losses, label="Training")
    ax[0].plot(epochs, val_losses, label="Validation")
    ax[0].set_title("Loss")
    ax[0].set(xlabel="Epochs", ylabel="Loss")
    ax[0].legend()

    ax[1].plot(epochs, train_accs, label="Training")
    ax[1].plot(epochs, val_accs, label="Validation")
    ax[1].set_title("Accuracy")
    ax[1].set(xlabel="Epochs", ylabel="Accuracy")
    ax[1].legend()

    f.suptitle(f'{model_name} - Loss/Accuracy Curves - {str(len(epochs))} Epochs')
    plt.savefig("imgs/metrics/" + model_name + "_" + str(len(epochs)) + "epochs" + ".png")
    plt.show()
