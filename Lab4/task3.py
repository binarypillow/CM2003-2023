import os
import sys
import yaml
import pickle
import numpy as np

# remove message about instructions in performance-critical operations: AVX2 FMA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

sys.path.append("../")
from utils import DataLoader, CompileFit, Plots, Models

with open(sys.argv[1], "r") as ymlfile:
    config = yaml.safe_load(ymlfile)

# Read config values
task_name = config["task_name"]
iter_name = config["iter_name"]
load_weights_history = config["load_weights_history"]
training_data_path = f"../../Data/{config['training_data_path']}"
validation_data_path = f"../../Data/{config['validation_data_path']}"

data = config["data"]
img_ch = data["img_ch"]
img_w = data["img_w"]
img_h = data["img_h"]
net = config["net"]
generator = config["generator"]

# Training of the model with data augmentation
train_generator = DataLoader.generateData(
    training_data_path,
    target_size=(img_w, img_h),
    batch_size=generator["batch_s"],
    class_mode=generator["class_mode"],
    shuffle=generator["shuffle"],
    rescale=eval(generator["rescale"]),
)
validation_generator = DataLoader.generateData(
    training_data_path,
    target_size=(img_w, img_h),
    batch_size=generator["batch_s"],
    class_mode=generator["class_mode"],
    shuffle=generator["shuffle"],
    rescale=eval(generator["rescale"]),
)

model = Models.VGG16(
    img_ch,
    img_w,
    img_h,
    base=net["base"],
    base_dense_1=net["base_dense_1"],
    dropout=net["dropout"],
    dropout_rate=net["dropout_rate"],
    output_layers=net["output_layers"],
    extra_dense=True,
)
# Compile and train the model, plot learning curves
if not load_weights_history:
    model_fit = CompileFit.compileFit(
        model,
        loss=net["loss"],
        opt=net["optimizer"],
        metrics=net["metrics"],
        train_generator=train_generator,
        learning_rate=net["learning_rate"],
        epochs=net["n_epoch"],
        valid_generator=validation_generator,
        batch_size=net["batch_s"],
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
Plots.plotLossAccuracy(history, task_name, iter_name, metric=net["metrics"])

sample_path = f"../../Data/{config['sample_path']}"
Plots.activationMapsPlot(
    sample_path, model, net["base"], img_w, img_h, task_name, iter_name
)
