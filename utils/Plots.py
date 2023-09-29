import matplotlib.pyplot as plt
import numpy as np
from tensorflow.math import confusion_matrix
from sklearn.metrics import roc_curve
import seaborn as sns


# loss and accuracy plots combined
def plotLossAccuracy(history, task_name="", iter_name=""):
    """This function shows a figure with two subplots (loss and accuracy) and optionally saves it.

    Args:
        history (object): the history object of the trained model
        task_name (str, optional): the folder in which the results are saved. If left blank, the image is only shown. Defaults to "".
        iter_name (str, optional): an optional extention to the default file name. Defaults to "".
    """

    fig, (ax1, ax2) = plt.subplots(2)

    fig.set_size_inches(17, 9)
    fig.tight_layout(pad=5.0)
    # Plot loss
    ax1.set_title("Loss")
    ax1.plot(history["loss"], label="Train")
    ax1.plot(history["val_loss"], label="Validation")
    ax1.plot(
        np.argmin(history["val_loss"]),
        np.min(history["val_loss"]),
        marker="x",
        color="r",
        label="best model",
    )
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    # Plot accuracy
    ax2.set_title("Accuracy")
    ax2.plot(history["binary_accuracy"], label="Train")
    ax2.plot(history["val_binary_accuracy"], label="Validation")
    ax2.plot(
        np.argmax(history["val_binary_accuracy"]),
        np.max(history["val_binary_accuracy"]),
        marker="x",
        color="r",
        label="best model",
    )
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    if task_name:
        plt.savefig(f"results/{task_name}/loss-accuracy{iter_name}.png")
    plt.show()


# confusion matrix
def plotCM(y_true, y_pred, task_name="", iter_name=""):
    """This function shows a CM plot and optionally saves it

    Args:
        y_test (array): an array with the true values.
        y_pred (array): an array with the predicted values.
        task_name (str, optional): the folder in which the results are saved. If left blank, the image is only shown. Defaults to "".
        iter_name (str, optional): an optional extention to the default file name. Defaults to "".
    """
    # cm with confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(
        np.eye(2),
        annot=cm,
        fmt="g",
        annot_kws={"size": 30},
        cmap=sns.color_palette(["tomato", "palegreen"], as_cmap=True),
        cbar=False,
        yticklabels=["True", "False"],
        xticklabels=["True", "False"],
    )
    plt.xlabel("Predicted values")
    plt.ylabel("True values")
    plt.title("Confusion Matrix (test set)")
    if task_name:
        plt.savefig(f"results/{task_name}/CM{iter_name}.png")
    plt.show()


# ROC plot
def plotROC(y_test, y_pred, task_name="", iter_name=""):
    """This function shows a ROC plot and optionally saves it.

    Args:
        y_test (array): an array with the true values.
        y_pred (array): an array with the predicted values.
        task_name (str, optional): the folder in which the results are saved. If left blank, the image is only shown. Defaults to "".
        iter_name (str, optional): an optional extention to the default file name. Defaults to "".
    """

    # Calculate FPR, TPR using NumPy
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    plt.figure(figsize=(5, 5))
    # Plot ROC curve using Matplotlib
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    if task_name:
        plt.savefig(f"results/{task_name}/ROC{iter_name}.png")
    plt.show()


# loss plot
def lossPlot(model_fit, task_name="", iter_name=""):
    """This function shows a loss plot and optionally saves it.

    Args:
        model_fit (object): the trained model
        task_name (str, optional): the folder in which the results are saved. If left blank, the image is only shown. Defaults to "".
        iter_name (str, optional): an optional extention to the default file name. Defaults to "".
    """

    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(model_fit.history["loss"], label="loss")
    plt.plot(model_fit.history["val_loss"], label="val_loss")
    plt.plot(
        np.argmin(model_fit.history["val_loss"]),
        np.min(model_fit.history["val_loss"]),
        marker="x",
        color="r",
        label="best model",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()
    if task_name:
        plt.savefig(f"results/{task_name}/loss{iter_name}.png")
    plt.show()


# accuracy plot
def accuracyPlot(model_fit, task_name="", iter_name=""):
    """This function shows an accuracy plot and optionally saves it.

    Args:
        model_fit (object): the trained model
        task_name (str, optional): the folder in which the results are saved. If left blank, the image is only shown. Defaults to "".
        iter_name (str, optional): an optional extention to the default file name. Defaults to "".
    """

    plt.figure(figsize=(4, 4))
    plt.title("Accuracy curve")
    plt.plot(model_fit.history["binary_accuracy"], label="binary_accuracy")
    plt.plot(model_fit.history["val_binary_accuracy"], label="val_binary_accuracy")
    plt.plot(
        np.argmax(model_fit.history["val_binary_accuracy"]),
        np.max(model_fit.history["val_binary_accuracy"]),
        marker="x",
        color="r",
        label="best model",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy Value")
    plt.legend()
    if task_name:
        plt.savefig(f"results/{task_name}/accuracy{iter_name}.png")
    plt.show()
