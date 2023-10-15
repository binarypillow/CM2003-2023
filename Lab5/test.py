import pickle
import numpy as np

with open(f"results/task3/history.pkl", "rb") as file_pi:
    historyy = pickle.load(file_pi)

history = historyy[0]
loss = history["loss"]
acc = history["dice_coeff"]
best_model_idx = np.argmax(acc)
print(loss[best_model_idx])
print(max(acc))
loss = history["val_loss"]
acc = history["val_dice_coeff"]
best_model_idx = np.argmax(acc)
print(loss[best_model_idx])
print(max(acc))
