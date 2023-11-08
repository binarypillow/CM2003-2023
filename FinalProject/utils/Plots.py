import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
from matplotlib.colors import ListedColormap
import tensorflow as tf
import cv2

tf.compat.v1.disable_eager_execution()

"""
----------------------
    LOSS ACCURACY
----------------------
"""


# loss and accuracy plots combined
def plotLossAccuracy(history, task_name="", iter_name="", metric=""):
    """This function shows a figure with three subplots (loss and accuracy) and optionally saves it

    Args:
        history (object): the history object of the trained model
        task_name (str, optional): the folder in which the results are saved. If left blank, the image is only shown. Defaults to "".
        iter_name (str, optional): an optional extention to the default file name. Defaults to "".
        metric (str, optional): the metric function to use. Defaults to "binary_accuracy".
    """

    fig, (ax1, ax2, ax3) = plt.subplots(3)

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

    # Plot Mean Dice score
    param = metric[0]
    ax2.set_title("Dice coefficient multilabel")
    ax2.set_ylabel("Dice coeff multi")
    ax2.plot(history[param], label="Train")
    ax2.plot(history["val_" + param], label="Validation")

    ax2.plot(
        np.argmax(history["val_" + param]),
        np.max(history["val_" + param]),
        marker="x",
        color="r",
        label="best model",
    )
    ax2.set_xlabel("Epoch")
    ax2.legend()

    # Plot Mean Jaccard score
    param = metric[3]
    ax3.set_title("Jaccard coefficient multilabel")
    ax3.set_ylabel("Jaccard coeff multi")
    ax3.plot(history[param], label="Train")
    ax3.plot(history["val_" + param], label="Validation")

    ax3.plot(
        np.argmax(history["val_" + param]),
        np.max(history["val_" + param]),
        marker="x",
        color="r",
        label="best model",
    )
    ax3.set_xlabel("Epoch")
    ax3.legend()

    if task_name:
        plt.savefig(f"results/{task_name}/{iter_name}/loss-dice-jaccard.png")
    plt.show()
    plt.close()


# loss and accuracy plots combined for cross validation
def plotLossAccuracyCV(history, n_folds, task_name="", iter_name="", metric=""):
    """This function shows a figure with three subplots (loss and accuracy) and optionally saves it

    Args:
        history (object): the history object of the trained model
        task_name (str, optional): the folder in which the results are saved. If left blank, the image is only shown. Defaults to "".
        iter_name (str, optional): an optional extention to the default file name. Defaults to "".
        metric (str, optional): the metric function to use. Defaults to "binary_accuracy".
    """

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.set_size_inches(17, 9)
    fig.tight_layout(pad=5.0)

    for fold in range(n_folds):
        # Plot loss
        ax1.plot(history[fold]["val_loss"], label=f"fold {fold} as valid")
        ax1.plot(
            np.argmin(history[fold]["val_loss"]),
            np.min(history[fold]["val_loss"]),
            marker="x",
            color="r",
        )

        # Plot Mean Dice score
        ax2.plot(
            history[fold]["val_dice_coeff_multilabel"], label=f"fold {fold} as valid"
        )
        ax2.plot(
            np.argmax(history[fold]["val_dice_coeff_multilabel"]),
            np.max(history[fold]["val_dice_coeff_multilabel"]),
            marker="x",
            color="r",
        )

        # Plot Mean Jaccard score
        ax3.plot(
            history[fold]["val_jaccard_coeff_multilabel"], label=f"fold {fold} as valid"
        )
        ax3.plot(
            np.argmax(history[fold]["val_jaccard_coeff_multilabel"]),
            np.max(history[fold]["val_jaccard_coeff_multilabel"]),
            marker="x",
            color="r",
        )

    ax1.set_title("CrossValidation Loss")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.set_title("Dice coefficient multilabel")
    ax2.set_ylabel("Dice coeff multi")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    ax3.set_title("Jaccard coefficient multilabel")
    ax3.set_ylabel("Jaccard coeff multi")
    ax3.set_xlabel("Epoch")
    ax3.legend()

    if task_name:
        plt.savefig(f"results/{task_name}/{iter_name}/loss-dice-jaccard_CV.png")
    plt.show()
    plt.close()


"""
---------------------
    SEGMENTATION
---------------------
"""


# segmentation plot
def plotSegmentationAV(
    img_list,
    img_path,
    mask_list,
    model,
    task_name="",
    iter_name="",
    output_layers=1,
    n_crops=0,
):
    """Compare ground truth segmentation with predicted one with a graph

    Args:
        img_list (array): array of the images.
        mask_path (string): path to the mask of the image.
        mask_list (array): array of masks.
        model (object): trained model.
        task_name (str, optional): the folder in which the results are saved. If left blank, the image is only shown. Defaults to "".
        iter_name (str, optional): an optional extention to the default file name. Defaults to "".
    """
    os.makedirs(f"results/{task_name}/{iter_name}/segmentation/", exist_ok=True)

    img_list_name = sorted(os.listdir(img_path))
    if n_crops:
        img_list_name_crop = []
        for img_name in img_list_name:
            for j in range(n_crops):
                img_list_name_crop.append(f"{img_name[:-4]}_{j}.png")
        img_list_name = img_list_name_crop

    for img, mask, img_name in zip(img_list, mask_list, img_list_name):
        pred = model.predict(np.expand_dims(img, axis=0))
        pred = pred[0]
        if output_layers != 1:
            mask = np.argmax(mask, axis=2)
            pred = np.argmax(pred, axis=2)
            pred = np.divide(pred.astype(np.float32), 2)
            pred[pred == 0.5] = 128  # arteries
            pred[pred == 1] = 256  # veins

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))

        fig.set_size_inches(17, 9)
        fig.tight_layout(pad=5.0)

        AVcmap = ListedColormap(["white", "red", "blue"])

        ax1.imshow(img, cmap="gray")
        ax1.set_title("Original Image")
        ax2.imshow(mask, cmap=AVcmap, interpolation="none")
        ax2.set_title("Ground truth")
        ax3.imshow(pred, cmap=AVcmap, interpolation="none")
        ax3.set_title("Prediction")

        if task_name:
            plt.savefig(f"results/{task_name}/{iter_name}/segmentation/{img_name}")
        plt.show()
        plt.close()


# save a figure with predicted segmentation
def saveSegmentationAV(
    img_list,
    img_path,
    model,
    task_name="",
    iter_name="",
    output_layers=1,
    n_crops=0,
    test_set=False,
):
    """Save the predicted segmentation in a file in the format requested by the challenge

    Args:
        img_list (array): array of the images.
        mask_path (string): path to the mask of the image.
        model (object): trained model.
    """
    os.makedirs(f"results/{task_name}/{iter_name}/prediction/", exist_ok=True)
    os.makedirs(f"results/{task_name}/{iter_name}/prediction/test/", exist_ok=True)

    img_list_name = sorted(os.listdir(img_path))
    if n_crops and not test_set:
        img_list_name_crop = []
        for img_name in img_list_name:
            for j in range(n_crops):
                img_list_name_crop.append(f"{img_name[:-4]}_{j}.png")
        img_list_name = img_list_name_crop
    img_preds = []
    for img, img_name in zip(img_list, img_list_name):
        pred = model.predict(np.expand_dims(img, axis=0))
        img_preds.append(pred)
        pred = pred[0]
        if output_layers != 1:
            pred = np.argmax(pred, axis=2)
            pred = np.divide(pred.astype(np.float32), 2)
            pred[pred == 0.5] = 128  # arteries
            pred[pred == 1] = 256  # veins
        if task_name:
            if not test_set:
                cv2.imwrite(
                    f"results/{task_name}/{iter_name}/prediction/{img_name}", pred
                )
            else:
                cv2.imwrite(
                    f"results/{task_name}/{iter_name}/prediction/test/{img_name}", pred
                )
    return np.array(img_preds)
