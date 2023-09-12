import matplotlib.pyplot as plt
import numpy as np

def lossPlot(model_fit):
    '''
    Parameters
    ----------
    model_fit : Object
    Trained model.
    '''
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(model_fit.history["loss"], label="loss")
    plt.plot(model_fit.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(model_fit.history["val_loss"]),
    np.min(model_fit.history["val_loss"]),
    marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.show()
    
def accuracyPlot(model_fit):
    '''
    Parameters
    ----------
    model_fit : Object
    Trained model.
    '''
    plt.figure(figsize=(4, 4))
    plt.title("Accuracy curve")
    plt.plot(model_fit.history["binary_accuracy"], label="binary_accuracy")
    plt.plot(model_fit.history["val_binary_accuracy"], label="val_binary_accuracy")
    plt.plot( np.argmax(model_fit.history["val_binary_accuracy"]),
    np.max(model_fit.history["val_binary_accuracy"]),
    marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy Value")
    plt.legend()
    plt.show()
