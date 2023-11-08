import os
import sys
import yaml
import numpy as np
from sklearn.model_selection import KFold
from keras.optimizers import Adam

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# import utils files
sys.path.append("../")
from FinalProject.utils import Models, DataLoader, Plots, LossMetrics

"""
--------------------
    LOAD CONFIG
--------------------
"""

with open(f"configs/tuning/{sys.argv[1]}", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)

# folder names
task_name = config["task_name"]
iter_name = config["iter_name"]
# create folders
os.makedirs(f"results/{task_name}/{iter_name}/", exist_ok=True)

# dataset paths
img_data_path = f"{config['img_data_path']}"
mask_data_path = f"{config['mask_data_path']}"
img_test_data_path = config["img_test_data_path"]

# image properties
data = config["data"]
img_ch = data["img_ch"]
img_w = data["img_w"]
img_h = data["img_h"]

# cross validation
n_folds = config["n_folds"]

# model properties
net = config["net"]

"""
---------------------
    DEFINE METRICS
---------------------
"""

# define a customised loss function
if net["loss"] == "dice_loss":
    loss = [LossMetrics.dice_loss]
else:
    loss = net["loss"]

# define a customised metric functions
metrics = []
for metric in net["metrics"]:
    if metric == "dice_coeff_multilabel":
        metrics.append(LossMetrics.dice_coeff_multilabel)
    elif metric == "dice_coeff_arteries":
        metrics.append(LossMetrics.dice_coeff_arteries)
    elif metric == "dice_coeff_veins":
        metrics.append(LossMetrics.dice_coeff_veins)
    elif metric == "jaccard_coeff_multilabel":
        metrics.append(LossMetrics.jaccard_coeff_multilabel)
    elif metric == "jaccard_coeff_arteries":
        metrics.append(LossMetrics.jaccard_coeff_arteries)
    elif metric == "jaccard_coeff_veins":
        metrics.append(LossMetrics.jaccard_coeff_veins)
    else:
        metrics.append(metric)

"""
--------------------
    LOAD DATASET
--------------------
"""

# load of data
images = DataLoader.loadDataImg(img_w, img_h, img_ch, img_data_path)
masks = DataLoader.loadDataMask(img_w, img_h, mask_data_path, binary=False)

"""
----------------------
    COMPILE AND FIT
----------------------
"""

folds_history = []
kf = KFold(n_folds)
for fold, (train_idx, valid_idx) in enumerate(kf.split(images)):
    # create train and validation from
    x_train = images[train_idx, :, :, :]
    y_train = masks[train_idx, :, :, :]
    x_valid = images[valid_idx, :, :, :]
    y_valid = masks[valid_idx, :, :, :]

    # define the model at each cycle
    model = Models.UNet(
        img_ch,
        img_w,
        img_h,
        base=net["base"],
        output_layers=net["output_layers"],
        dropout=net["dropout"],
        dropout_rate=net["dropout_rate"],
        normalization=net["normalization"],
    )
    model.summary()

    # compile the model
    model.compile(
        loss=loss,
        optimizer=Adam(net["learning_rate"]),
        metrics=metrics,
    )

    model_fit = model.fit(
        x_train,
        y_train,
        batch_size=net["batch_s"],
        epochs=net["n_epoch"],
        validation_data=[x_valid, y_valid],
        verbose=1,
    )

    folds_history.append(model_fit.history)

# generate plots and save figures
Plots.plotLossAccuracyCV(
    folds_history, n_folds, task_name, iter_name, metric=net["metrics"]
)

"""
---------------------------------
    CALCULATE TRAINING METRICS
---------------------------------
"""

mean_metrics = {}
folds_metric = []
for metric in net["metrics"]:
    folds_metric = [np.max(folds_history[n][f"val_{metric}"]) for n in range(n_folds)]
    mean_metrics[f"Mean val_{metric}"] = float(np.mean(folds_metric))
    mean_metrics[f"Std val_{metric}"] = float(np.std(folds_metric))

# store max metrics
config["mean_metrics"] = mean_metrics

# print the results
with open(f"configs/tuning/{sys.argv[1]}", "w") as yaml_file:
    yaml.dump(config, yaml_file, default_flow_style=False, sort_keys=False)
