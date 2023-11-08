import os
import sys
import yaml
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# import utils files
sys.path.append("../")
from FinalProject.utils import Models, DataLoader, Plots, LossMetrics

"""
--------------------
    LOAD CONFIG
--------------------
"""

with open(f"configs/{sys.argv[1]}", "r") as ymlfile:
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

# LR scheduler
use_LR_scheduler = config["use_LR_scheduler"]

# random crop
use_random_crop = config["use_random_crop"]
n_crops = config["n_crops"]
img_w_crop = data["img_w_crop"]
img_h_crop = data["img_h_crop"]

# learning transfer
use_transfer = config["use_transfer"]

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
images_original = DataLoader.loadDataImg(img_w, img_h, img_ch, img_data_path)
masks_original = DataLoader.loadDataMask(img_w, img_h, mask_data_path, binary=False)

# implement random cropping, if required
if use_random_crop:
    images, masks = DataLoader.get_randomCrop(
        images_original,
        masks_original,
        img_w_crop,
        img_h_crop,
        img_ch,
        n_crops,
    )
else:
    images, masks = images_original, masks_original

if use_transfer:
    images = np.repeat(images, 3, -1)  # the loaded weights work only with RGB images

"""
----------------------
    COMPILE AND FIT
----------------------
"""


# define learning rate scheduler
def lr_schedule(epoch):
    initial_lr = 0.001
    decay_factor = 0.5
    epoch_per_decay = 100

    # halve the learning rate each 100 epochs
    lr = initial_lr * (decay_factor ** (epoch // epoch_per_decay))
    return lr


lr_scheduler = LearningRateScheduler(lr_schedule)

x_train, x_valid, y_train, y_valid = train_test_split(
    images, masks, test_size=0.20, random_state=42
)

if use_transfer:
    model = Models.UNetVGG16(
        3,
        img_w,
        img_h,
        base=net["base"],
        output_layers=net["output_layers"],
        dropout=net["dropout"],
        dropout_rate=net["dropout_rate"],
        normalization=net["normalization"],
    )
else:
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

if use_LR_scheduler:
    model_fit = model.fit(
        x_train,
        y_train,
        batch_size=net["batch_s"],
        epochs=net["n_epoch"],
        validation_data=[x_valid, y_valid],
        verbose=1,
        callbacks=[lr_scheduler],
    )
else:
    model_fit = model.fit(
        x_train,
        y_train,
        batch_size=net["batch_s"],
        epochs=net["n_epoch"],
        validation_data=[x_valid, y_valid],
        verbose=1,
    )

history = model_fit.history


# generate plot
Plots.plotLossAccuracy(history, task_name, iter_name, metric=net["metrics"])

"""
---------------------------------
    CALCULATE TRAINING METRICS
---------------------------------
"""

best_metrics = {}
for metric in net["metrics"]:
    best_metrics[f"Max val_{metric}"] = float(np.max(history[f"val_{metric}"]))
# store max metrics
config["best_metrics"] = best_metrics

"""
------------------------
    PLOT PREDICTIONS
------------------------
"""

# save segmentation comparisons
Plots.plotSegmentationAV(
    images,
    img_data_path,
    masks,
    model,
    task_name,
    iter_name,
    output_layers=net["output_layers"],
    n_crops=n_crops,
)

if use_transfer:
    images_original = np.repeat(images_original, 3, -1)

# save predicted segmentations
img_pred = Plots.saveSegmentationAV(
    images_original,
    img_data_path,
    model,
    task_name=task_name,
    iter_name=iter_name,
    output_layers=net["output_layers"],
)

# print final metrics values
dice_coeff_multilabel = []
dice_coeff_arteries = []
dice_coeff_veins = []
jaccard_coeff_multilabel = []
jaccard_coeff_arteries = []
jaccard_coeff_veins = []

# calculate mean dice + std dice and mean jaccard + std jaccard
for mask, pred in zip(masks_original, img_pred):
    mask = np.expand_dims(mask, axis=0)
    with tf.compat.v1.Session() as session:
        if "dice_coeff_multilabel" in net["metrics"]:
            dice_coeff_multilabel.append(
                session.run(LossMetrics.dice_coeff_multilabel(mask, pred))
            )
        if "dice_coeff_arteries" in net["metrics"]:
            dice_coeff_arteries.append(
                session.run(LossMetrics.dice_coeff_arteries(mask, pred))
            )
        if "dice_coeff_veins" in net["metrics"]:
            dice_coeff_veins.append(
                session.run(LossMetrics.dice_coeff_veins(mask, pred))
            )
        if "jaccard_coeff_multilabel" in net["metrics"]:
            jaccard_coeff_multilabel.append(
                session.run(LossMetrics.jaccard_coeff_multilabel(mask, pred))
            )
        if "jaccard_coeff_arteries" in net["metrics"]:
            jaccard_coeff_arteries.append(
                session.run(LossMetrics.jaccard_coeff_arteries(mask, pred))
            )
        if "jaccard_coeff_veins" in net["metrics"]:
            jaccard_coeff_veins.append(
                session.run(LossMetrics.jaccard_coeff_veins(mask, pred))
            )

results = {}

for metric_name, metric_list in {
    "dice_coeff_multilabel": dice_coeff_multilabel,
    "dice_coeff_arteries": dice_coeff_arteries,
    "dice_coeff_veins": dice_coeff_veins,
    "jaccard_coeff_multilabel": jaccard_coeff_multilabel,
    "jaccard_coeff_arteries": jaccard_coeff_arteries,
    "jaccard_coeff_veins": jaccard_coeff_veins,
}.items():
    if metric_name in net["metrics"]:
        results[f"Mean {metric_name}"] = float(np.mean(metric_list))
        results[f"Std {metric_name}"] = float(np.std(metric_list))

config["results"] = results

# print the results
with open(f"configs/{sys.argv[1]}", "w") as yaml_file:
    yaml.dump(config, yaml_file, default_flow_style=False, sort_keys=False)

"""
---------------------------
    TEST SET PREDICTIONS
---------------------------
"""

# load test set images
images_test = DataLoader.loadDataImg(img_w, img_h, img_ch, img_test_data_path)

if use_transfer:
    images_test = np.repeat(images_test, 3, -1)

# save predicted segmentations
_ = Plots.saveSegmentationAV(
    images_test,
    img_test_data_path,
    model,
    task_name=task_name,
    iter_name=iter_name,
    output_layers=net["output_layers"],
    test_set=True,
)
