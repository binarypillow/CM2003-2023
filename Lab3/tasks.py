import os
import sys
import yaml
import pickle
import numpy as np

# remove message about instructions in performance-critical operations: AVX2 FMA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

sys.path.append("../")
from utils import DataLoader, Models, CompileFit, Plots

with open(sys.argv[1], "r") as ymlfile:
    config = yaml.safe_load(ymlfile)

task_name = config["task_name"]
iter_name = config["iter_name"]
load_weights_history = config["load_weights_history"]
data_path = config["data_path"]
use_preloaded_data = config["use_preloaded_data"]

data = config["data"]
net = config["net"]
img_ch = data["img_ch"]
img_w = data["img_w"]
img_h = data["img_h"]

# load data
if not use_preloaded_data:
    x_train, x_test, y_train, y_test = DataLoader.loadData(img_w, img_h, data_path)
    with open(f"data/{data_path.replace('/','')}.pkl", "wb") as file_pi:
        pickle.dump(x_train, file_pi)
        pickle.dump(x_test, file_pi)
        pickle.dump(y_train, file_pi)
        pickle.dump(y_test, file_pi)
else:
    with open(f"data/{data_path.replace('/','')}.pkl", "rb") as file_pi:
        x_train = pickle.load(file_pi)
        x_test = pickle.load(file_pi)
        y_train = pickle.load(file_pi)
        y_test = pickle.load(file_pi)


# create model
net_model = net["net_model"]
if net_model == "VGG16":
    model = Models.VGG16(
        img_ch=img_ch,
        img_width=img_w,
        img_height=img_h,
        base=net["base"],
        base_dense_1=net["base_dense_1"],
        dropout=net["dropout"],
        dropout_rate=net["dropout_rate"],
        output_layers=net["output_layers"],
    )
elif net_model == "AlexNet":
    model = Models.AlexNet(
        img_ch=img_ch,
        img_width=img_w,
        img_height=img_h,
        base=net["base"],
        base_dense_1=net["base_dense_1"],
        base_dense_2=net["base_dense_2"],
        normalization=net["normalization"],
        spatial_drop=net["spatial_drop"],
        spatial_drop_rate=net["spatial_drop_rate"],
        dropout=net["dropout"],
        dropout_rate=net["dropout_rate"],
        output_layers=net["output_layers"],
    )
elif net_model == "LeNet":
    model = Models.LeNet(
        img_ch=img_ch,
        img_width=img_w,
        img_height=img_h,
        base=net["base"],
        base_dense_1=net["base_dense_1"],
        dropout=net["dropout"],
        dropout_rate=net["dropout_rate"],
        output_layers=net["output_layers"],
    )

if not load_weights_history:
    # compile and fit model
    model_fit = CompileFit.compileFit(
        model=model,
        loss="BinaryCrossentropy",
        opt=net["optimizer"],
        metrics=net["metrics"],
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
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
predictions = model.predict(x_test)

# generate plots and save figures
Plots.plotLossAccuracy(history, task_name, iter_name)
Plots.plotROC(y_test, predictions, task_name, iter_name)

if net["output_layers"] < 2:
    y_test = [np.round(item[0]) for item in y_test]
    predictions = [np.round(item[0]) for item in predictions]
    Plots.plotCM(y_test, predictions, task_name, iter_name)
