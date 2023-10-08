import os
import sys
import yaml
import pickle
import numpy as np

# remove message about instructions in performance-critical operations: AVX2 FMA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

sys.path.append("../")
from utils import DataLoader, CompileFit, Plots

with open(sys.argv[1], "r") as ymlfile:
    config = yaml.safe_load(ymlfile)

from tensorflow.keras import applications, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

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

# number of data for each class
Len_C1_Train = DataLoader.get_length(training_data_path, data["cat1"])
Len_C2_Train = DataLoader.get_length(training_data_path, data["cat2"])
Len_C1_Val = DataLoader.get_length(validation_data_path, data["cat1"])
Len_C2_Val = DataLoader.get_length(validation_data_path, data["cat2"])

# Loading the pre-trained model
# include top: false means that the dense layers at the top of the network will not be used
model = applications.VGG16(include_top=False, weights="imagenet")
model.summary()
# Feature extraction from pretrained VGG (training data)
train_generator = DataLoader.generateData(
    training_data_path,
    target_size=(img_w, img_h),
    batch_size=generator["batch_s"],
    class_mode=generator["class_mode"],
    shuffle=generator["shuffle"],
    rescale=eval(generator["rescale"]),
)

# Extracting the features from the loaded images
features_train = model.predict(
    train_generator, (Len_C1_Train + Len_C2_Train) // net["batch_s"], max_queue_size=1
)

# Feature extraction from pretrained VGG (validation data)
validation_generator = DataLoader.generateData(
    validation_data_path,
    target_size=(img_w, img_h),
    batch_size=generator["batch_s"],
    class_mode=generator["class_mode"],
    shuffle=generator["shuffle"],
    rescale=eval(generator["rescale"]),
)
# Extracting the features from the loaded images
features_validation = model.predict(
    validation_generator, (Len_C1_Val + Len_C2_Val) // net["batch_s"], max_queue_size=1
)


# Training a small MLP with extracted features from the pre-trained model
# In fact this MLP will be used instead of the dense layers of the VGG model
# and only this MLP will be trained on the dataset.
train_data = features_train
train_labels = np.array([0] * int(Len_C1_Train) + [1] * int(Len_C2_Train))

validation_data = features_validation
validation_labels = np.array([0] * int(Len_C1_Val) + [1] * int(Len_C2_Val))


# Building the MLP model
# create model
net_model = net["net_model"]
if net_model == "MLP":

    def MLP():
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(net["base_dense_1"], activation="relu"))
        model.add(Dropout(net["dropout_rate"]))
        model.add(Dense(net["output_layers"], activation="sigmoid"))

        return model

    model = MLP()


# Compile and train the model, plot learning curves
if not load_weights_history:
    model_fit = CompileFit.compileFit(
        model,
        loss=net["loss"],
        opt=net["optimizer"],
        metrics=net["metrics"],
        x_train=train_data,
        y_train=train_labels,
        x_test=validation_data,
        y_test=validation_labels,
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
Plots.plotLossAccuracy(history, task_name, iter_name)
