import matplotlib.pyplot as plt
import numpy as np
from tensorflow.math import confusion_matrix
from sklearn.metrics import roc_curve
import seaborn as sns
from tensorflow.keras import backend as K
import cv2
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


# loss and accuracy plots combined
def plotLossAccuracy(history, task_name="", iter_name="", metric=""):
    """This function shows a figure with two subplots (loss and accuracy) and optionally saves it.

    Args:
        history (object): the history object of the trained model
        task_name (str, optional): the folder in which the results are saved. If left blank, the image is only shown. Defaults to "".
        iter_name (str, optional): an optional extention to the default file name. Defaults to "".
        metric (str, optional): the metric function to use. Defaults to "binary_accuracy.
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

    if metric == "":
        param = "binary_accuracy"
        ax2.set_title("Accuracy")
        ax2.set_ylabel("Accuracy")
    else:
        param = metric
        ax2.set_title(metric)
        ax2.set_ylabel(metric)

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


# activation maps plot
def activationMapsPlot(
    sample_path, model, base, img_w, img_h, task_name="", iter_name=""
):
    """This function shows an activation map and optionally saves it.

    Args:
        sample_path (string): path to the image.
        model (object): the trained model.
        base (int): the number of neurons of the dense layer.
        img_w (int): the image width.
        img_h (int): the image height.
        task_name (str, optional): the folder in which the results are saved. If left blank, the image is only shown. Defaults to "".
        iter_name (str, optional): an optional extention to the default file name. Defaults to "".
    """
    img = imread(sample_path)
    img = np.divide(img.astype(np.float32), 255.0)
    img = resize(img, (img_h, img_w), anti_aliasing=True).astype("float32")
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    class_idx = np.argmax(preds[0])
    print("The predicted class label is {}".format(class_idx))
    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer("Last_ConvLayer")
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img])
    for i in range(base * 8):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    img_ = cv2.imread(sample_path)
    img_ = cv2.resize(img_, (512, 512), interpolation=cv2.INTER_AREA)
    # img = img/255
    heatmap = cv2.resize(heatmap, (img_.shape[1], img_.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_, 0.6, heatmap, 0.4, 0)
    plt.figure()
    plt.imshow(img_)
    if task_name:
        plt.savefig(f"results/{task_name}/activation_maps{iter_name}.png")
    plt.figure()
    plt.imshow(superimposed_img)
    if task_name:
        plt.savefig(f"results/{task_name}/superimposed_activation_maps{iter_name}.png")
    plt.show()


# segmentation plot
def plotSegmentation(
    sample_img_path,
    sample_mask_path,
    model,
    img_w,
    img_h,
    img_ch,
    task_name="",
    iter_name="",
):
    """Compare ground truth segmentation with predicted one with a graph.

    Args:
        sample_img_path (string): path to the image.
        sample_mask_path (string): path to the mask of the image.
        model (object): trained model.
        img_w (int): the image width.
        img_h (int): the image height.
        img_ch (int): the image channels.
        task_name (str, optional): the folder in which the results are saved. If left blank, the image is only shown. Defaults to "".
        iter_name (str, optional): an optional extention to the default file name. Defaults to "".
    """
    img = imread(sample_img_path)
    img = np.divide(img.astype(np.float32), 255.0)
    img = resize(img, (img_h, img_w, img_ch), anti_aliasing=True).astype("float32")

    mask = imread(sample_mask_path)
    mask = np.divide(mask.astype(np.float32), 255.0)
    mask = resize(mask, (img_h, img_w, 1), anti_aliasing=True).astype("float32")

    pred = model.predict(np.expand_dims(img, axis=0))
    pred = pred[0]

    """
    prediction = model.predict(img[tf.newaxis, ...])[0]
    pred = (prediction > 0.5).astype(np.uint8)
    """

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))

    fig.set_size_inches(17, 9)
    fig.tight_layout(pad=5.0)

    ax1.imshow(img[:, :, 1], cmap="gray")
    ax1.set_title("Original Image")
    ax2.imshow(mask, cmap="gray")
    ax2.set_title("Ground truth")
    ax3.imshow(pred, cmap="gray")
    ax3.set_title("Prediction")

    if task_name:
        plt.savefig(f"results/{task_name}/segmentation_example{iter_name}.png")
    plt.show()
