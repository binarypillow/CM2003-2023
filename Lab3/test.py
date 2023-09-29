import pickle
import numpy as np

with open(f"results/task1b/history_05.pkl", "rb") as file_pi:
    history = pickle.load(file_pi)


loss = history["loss"]
acc = history["binary_accuracy"]
best_model_idx = np.argmax(acc)
print(loss[best_model_idx])
# print(acc.index(0.8349999785423279))
print(max(acc))
loss = history["val_loss"]
acc = history["val_binary_accuracy"]
best_model_idx = np.argmax(acc)
print(loss[best_model_idx])
# print(acc.index(0.9399999976158142))
print(max(acc))
