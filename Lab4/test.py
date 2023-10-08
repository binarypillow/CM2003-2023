import pickle
import numpy as np

with open(f"results/task4/history_e.pkl", "rb") as file_pi:
    history = pickle.load(file_pi)


loss = history["loss"]
acc = history["dice_coeff"]
best_model_idx = np.argmax(acc)
print(loss[best_model_idx])
# print(acc.index(0.8349999785423279))
print(max(acc))
loss = history["val_loss"]
acc = history["val_dice_coeff"]
best_model_idx = np.argmax(acc)
print(loss[best_model_idx])
# print(acc.index(0.9399999976158142))
print(max(acc))
