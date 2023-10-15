import os
import sys
import yaml
import pickle
import numpy as np
from tensorflow.keras.optimizers import Adam

# remove message about instructions in performance-critical operations: AVX2 FMA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

sys.path.append("../")
from utils import Models, DataLoader, Plots, LossMetrics, Resampling

with open(sys.argv[1], "r") as ymlfile:
    config = yaml.safe_load(ymlfile)

# read config values
task_name = config["task_name"]
iter_name = config["iter_name"]
load_weights_history = config["load_weights_history"]
load_data = config["load_data"]
data_path = f"../../Data/{config['data_path']}"
img_data_path = f"../../Data/{config['img_data_path']}"
mask_data_path = f"../../Data/{config['mask_data_path']}"
binary = config["binary"]

data = config["data"]
img_ch = data["img_ch"]
img_w = data["img_w"]
img_h = data["img_h"]
net = config["net"]

if net["loss"] == "dice_loss":
    loss = [LossMetrics.dice_loss]
else:
    loss = net["loss"]

if net["metrics"] == "dice_coeff":
    metrics = [LossMetrics.dice_coeff]
elif "dice_coeff" in net["metrics"]:
    metrics = [LossMetrics.dice_coeff]
    for i in range(1, len(net["metrics"])):
        metrics.append(net["metrics"][i])
else:
    metrics = net["metrics"]

if not load_data:
    # load of data
    images, masks = DataLoader.loadDataSeg(
        img_w, img_h, img_ch, img_data_path, mask_data_path, binary=binary
    )
    with open(
        f"data/MRI_{config['task_name']}{config['iter_name']}.pkl", "wb"
    ) as file_pi:
        pickle.dump(images, file_pi)
        pickle.dump(masks, file_pi)
else:
    with open(
        f"data/MRI_{config['task_name']}{config['iter_name']}.pkl", "rb"
    ) as file_pi:
        images = pickle.load(file_pi)
        masks = pickle.load(file_pi)
    print("Data Loaded!")

# Define the model
model = Models.UNet(
    img_ch,
    img_w,
    img_h,
    base=net["base"],
    output_layers=net["output_layers"],
    dropout=net["dropout"],
    dropout_rate=net["dropout_rate"],
    normalization=net["normalization"],
    auto_context=True,
)
model.summary()

model.compile(
    loss=loss,
    optimizer=Adam(learning_rate=net["learning_rate"]),
    metrics=metrics,
)

n_folds = config["n_folds"]
n_iter = config["num_ctx_iterations"]
# train the model
if not load_weights_history:
    history = []
    for i in range(n_folds):
        x_train, y_train, x_valid, y_valid = Resampling.k_fold(
            images, masks, n_folds, i
        )
        # Initialize predictions
        initial_prediction = np.zeros((x_train.shape[0], img_h, img_w, 1))
        train_autoctx_input = np.concatenate((x_train, initial_prediction), axis=-1)
        initial_prediction = np.zeros((x_valid.shape[0], img_h, img_w, 1))
        valid_autoctx_input = np.concatenate((x_valid, initial_prediction), axis=-1)
        for iteration in range(n_iter):
            # fit model
            model_fit = model.fit(
                x=[x_train, train_autoctx_input],
                y=y_train,
                validation_data=[[x_valid, valid_autoctx_input], y_valid],
                epochs=net["n_epoch"],
            )
            # Get and update the prediction for the next iteration
            prev_prediction = model.predict([x_train, train_autoctx_input])
            train_autoctx_input = np.concatenate((x_train, prev_prediction), axis=-1)
            prev_prediction = model.predict([x_valid, valid_autoctx_input])
            valid_autoctx_input = np.concatenate((x_valid, prev_prediction), axis=-1)
        history.append(model_fit.history)

    with open(f"results/{task_name}/history{iter_name}.pkl", "wb") as file_pi:
        pickle.dump(history, file_pi)
else:
    # load the history of the training process
    with open(f"results/{task_name}/history{iter_name}.pkl", "rb") as file_pi:
        history = pickle.load(file_pi)

fold_dice = []
# generate plots and save figures
for i in range(n_folds):
    Plots.plotLossAccuracy(
        history[i], task_name, f"{iter_name}_{i}", metric=["dice_coeff"]
    )
    fold_dice.append(np.max(history[i]["dice_coeff"]))

fold_mean = np.mean(fold_dice)
fold_std = np.std(fold_dice)
print(f"Dice coeff: {fold_mean:.3f}+-{fold_std:.3f}")
