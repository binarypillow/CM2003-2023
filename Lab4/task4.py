import os
import sys
import yaml
import pickle
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# remove message about instructions in performance-critical operations: AVX2 FMA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

sys.path.append("../")
from utils import Models, DataLoader, CompileFit, Plots

with open(sys.argv[1], "r") as ymlfile:
    config = yaml.safe_load(ymlfile)

# Read config values
task_name = config["task_name"]
iter_name = config["iter_name"]
load_weights_history = config["load_weights_history"]
data_path = f"../../Data/{config['data_path']}"
img_data_path = f"../../Data/{config['img_data_path']}"
mask_data_path = f"../../Data/{config['mask_data_path']}"
use_augmentation = config["use_augmentation"]

data = config["data"]
img_ch = data["img_ch"]
img_w = data["img_w"]
img_h = data["img_h"]
net = config["net"]
generator = config["generator"]

if net["loss"] == "dice_loss":
    loss = [CompileFit.dice_loss]
else:
    loss = net["loss"]

if net["metrics"] == "dice_coeff":
    metrics = [CompileFit.dice_coeff]
else:
    metrics = net["metrics"]

# Loading of data
if not use_augmentation:
    images, masks = DataLoader.loadDataSeg1(
        img_w, img_h, img_ch, img_data_path, mask_data_path
    )
    x_train, x_test, y_train, y_test = train_test_split(
        images, masks, test_size=0.20, random_state=42
    )

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

    fig.set_size_inches(17, 9)
    fig.tight_layout(pad=5.0)

    ax1.imshow(images[7, :, :, 1])
    ax1.set_title("Original Image")
    ax2.imshow(masks[7])
    ax2.set_title("Ground truth")
    plt.savefig(f"results/{task_name}/loadimage_example{iter_name}.png")
    """

else:
    img_train = DataLoader.generateData(
        data_path,
        target_size=(img_w, img_h),
        batch_size=generator["batch_s"],
        class_mode=generator["class_mode"],
        classes=["Image"],
        shuffle=generator["shuffle"],
        rescale=eval(generator["rescale"]),
        rotation_range=generator["rotation_range"],
        width_shift_range=generator["width_shift_range"],
        height_shift_range=generator["width_shift_range"],
        zoom_range=generator["zoom_range"],
        horizontal_flip=generator["horizontal_flip"],
        seed=generator["seed"],
        validation_split=0.2,
        subset="training",
    )
    img_valid = DataLoader.generateData(
        data_path,
        target_size=(img_w, img_h),
        batch_size=generator["batch_s"],
        class_mode=generator["class_mode"],
        classes=["Image"],
        shuffle=generator["shuffle"],
        rescale=eval(generator["rescale"]),
        rotation_range=generator["rotation_range"],
        width_shift_range=generator["width_shift_range"],
        height_shift_range=generator["width_shift_range"],
        seed=generator["seed"],
        validation_split=0.2,
        subset="validation",
    )
    mask_train = DataLoader.generateData(
        data_path,
        target_size=(img_w, img_h),
        batch_size=generator["batch_s"],
        color_mode=generator["color_mode"],
        class_mode=generator["class_mode"],
        classes=["Mask"],
        shuffle=generator["shuffle"],
        rescale=eval(generator["rescale"]),
        rotation_range=generator["rotation_range"],
        width_shift_range=generator["width_shift_range"],
        height_shift_range=generator["width_shift_range"],
        seed=generator["seed"],
        validation_split=0.2,
        subset="training",
    )
    mask_valid = DataLoader.generateData(
        data_path,
        target_size=(img_w, img_h),
        batch_size=generator["batch_s"],
        color_mode=generator["color_mode"],
        class_mode=generator["class_mode"],
        classes=["Mask"],
        shuffle=generator["shuffle"],
        rescale=eval(generator["rescale"]),
        rotation_range=generator["rotation_range"],
        width_shift_range=generator["width_shift_range"],
        height_shift_range=generator["width_shift_range"],
        seed=generator["seed"],
        validation_split=0.2,
        subset="validation",
    )

    train_generator = zip(img_train, mask_train)
    valid_generator = zip(img_valid, mask_valid)

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

if not load_weights_history and use_augmentation:
    # compile and fit model
    model_fit = CompileFit.compileFit(
        model=model,
        loss=loss,
        opt=net["optimizer"],
        metrics=metrics,
        learning_rate=net["learning_rate"],
        batch_size=net["batch_s"],
        epochs=net["n_epoch"],
        task_name=task_name,
        iter_name=iter_name,
        train_generator=train_generator,
        valid_generator=valid_generator,
        steps_per_epoch=net["step_per_epoch"],
        validation_steps=net["validation_step"],
    )
    history = model_fit.history

    with open(f"results/{task_name}/history{iter_name}.pkl", "wb") as file_pi:
        pickle.dump(history, file_pi)

elif not load_weights_history and not use_augmentation:
    # compile and fit model
    model_fit = CompileFit.compileFit(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        loss=loss,
        opt=net["optimizer"],
        metrics=metrics,
        learning_rate=net["learning_rate"],
        batch_size=net["batch_s"],
        epochs=net["n_epoch"],
        task_name=task_name,
        iter_name=iter_name,
    )
    history = model_fit.history

    with open(f"results/{task_name}/history{iter_name}.pkl", "wb") as file_pi:
        pickle.dump(history, file_pi)
else:
    # load the history of the training process
    with open(f"results/{task_name}/history{iter_name}.pkl", "rb") as file_pi:
        history = pickle.load(file_pi)

# load the saved best weights
model.load_weights(f"results/{task_name}/cp{iter_name}.ckpt")

# generate plots and save figures
Plots.plotLossAccuracy(history, task_name, iter_name, metric="dice_coeff")

# representation of image segmentation
sample_img_path = f"{img_data_path}/{config['sample']}"
sample_mask_path = f"{mask_data_path}/{config['sample']}"
Plots.plotSegmentation(
    sample_img_path,
    sample_mask_path,
    model,
    img_w,
    img_h,
    img_ch,
    task_name,
    iter_name,
)
