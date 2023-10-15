import os
import sys
import yaml
import pickle
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# remove message about instructions in performance-critical operations: AVX2 FMA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

sys.path.append("../")
from utils import Models, DataLoader, Plots, LossMetrics

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
use_augmentation = config["use_augmentation"]
binary = config["binary"]

data = config["data"]
img_ch = data["img_ch"]
img_w = data["img_w"]
img_h = data["img_h"]
net = config["net"]
generator = config["generator"]

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

# load of data
if not load_data:
    images, masks = DataLoader.loadDataSeg(
        img_w, img_h, img_ch, img_data_path, mask_data_path, binary=binary
    )

    with open(
        f"data/CT_{config['task_name']}{config['iter_name']}.pkl",
        "wb",
    ) as file_pi:
        pickle.dump(images, file_pi)
        pickle.dump(masks, file_pi)
else:
    with open(
        f"data/CT_{config['task_name']}{config['iter_name']}.pkl", "rb"
    ) as file_pi:
        images = pickle.load(file_pi)
        masks = pickle.load(file_pi)
        print("Data Loaded!")

x_train, x_test, y_train, y_test = train_test_split(
    images, masks, test_size=0.20, random_state=42
)

if use_augmentation:
    # define image generator
    datagen = ImageDataGenerator(
        rotation_range=generator["rotation_range"],
        width_shift_range=generator["width_shift_range"],
        height_shift_range=generator["width_shift_range"],
        zoom_range=generator["zoom_range"],
        horizontal_flip=generator["horizontal_flip"],
    )
    train_generator = datagen.flow(
        x_train,
        y=y_train,
        batch_size=generator["batch_s"],
        shuffle=generator["shuffle"],
        seed=generator["seed"],
    )
    valid_generator = datagen.flow(
        x_test,
        y=y_test,
        batch_size=generator["batch_s"],
        shuffle=generator["shuffle"],
        seed=generator["seed"],
    )

# define the model
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

model.compile(
    loss=loss,
    optimizer=Adam(net["learning_rate"]),
    metrics=metrics,
)

# train the model
if not load_weights_history and use_augmentation:
    # compile and fit model
    model_fit = model.fit(
        train_generator,
        epochs=net["n_epoch"],
        steps_per_epoch=net["step_per_epoch"],
        validation_steps=net["validation_step"],
        validation_data=valid_generator,
        verbose=1,
    )
    history = model_fit.history

    with open(f"results/{task_name}/history{iter_name}.pkl", "wb") as file_pi:
        pickle.dump(history, file_pi)

elif not load_weights_history and not use_augmentation:
    # compile and fit model
    model_fit = model.fit(
        x_train,
        y_train,
        batch_size=net["batch_s"],
        epochs=net["n_epoch"],
        validation_data=[x_test, y_test],
        verbose=1,
    )
    history = model_fit.history

    with open(f"results/{task_name}/history{iter_name}.pkl", "wb") as file_pi:
        pickle.dump(history, file_pi)
else:
    # load the history of the training process
    with open(f"results/{task_name}/history{iter_name}.pkl", "rb") as file_pi:
        history = pickle.load(file_pi)

# generate plots and save figures
Plots.plotLossAccuracy(history, task_name, metric=[net["metrics"]])

# representation of image segmentation
sample_img_path = f"{img_data_path}/{config['sample']}"
sample_mask_path = f"{mask_data_path}/{config['sample'].replace('.png','_Mask.png')}"

Plots.plotSegmentation(
    sample_img_path,
    sample_mask_path,
    model,
    img_w,
    img_h,
    img_ch,
    task_name,
    iter_name,
    binary=binary,
    output_layers=net["output_layers"],
)
