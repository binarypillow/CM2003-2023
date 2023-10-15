---
title: Report laboratory 5
author: Alessia Egano, Simone Bonino
---

# Report laboratory 5

⚠️ The complete results of this lab, such as saved training history, saved weights of the best model and graphs, can be found in the [results](/Lab5/results) folder.

## Task 1

### 1A)

**Reuse the same pipeline of task1 for segmentation of lung regions within the “CT” dataset and report the results**

![Learning curve and accuracy - Task 1A](/Lab5/results/task1/loss-accuracy_a.png)

|Accuracy  |Loss      |
|----------|----------|
|    0.9905|   -0.9905|

![Example of segmentation - Task 1A](/Lab5/results/task1/segmentation_example_a.png)

The predicted segmentation is almost perfect as shown in the figure above.

### 1B)

**Repeat the task 1a by including data augmentation technique (similar to task4e from Lab 4) and train the model for 50 epochs and interpret the observed values of the three metrics over the validation set. In general, what can be inferred from precision and recall metrics?**

![Learning curve and accuracy - Task 1B](/Lab5/results/task1/loss-accuracy_b_Dice.png)
![Learning curve and accuracy - Task 1B](/Lab5/results/task1/loss-accuracy_b_Recall.png)
![Learning curve and accuracy - Task 1B](/Lab5/results/task1/loss-accuracy_b_Precision.png)

|           |Accuracy  |Loss      |
|-----------|----------|----------|
|       Dice|    0.7793|   -0.7793|
|  Precision|    0.8391|   -0.5958|
|     Recall|    0.8761|   -0.7579|

The validation metrics show an overall decrease in performance after data augmentation, but identification of ROI remains good. The predicted segmented is less precise, but it is still able to identify the region in which the lungs are.

![Example of segmentation - Task 1B](/Lab5/results/task1/segmentation_example_b.png)

## Task 2

**Modify your model and adapt it for a multi-organ segmentation task and segment the left and right lungs separately in one framework. Choose a proper loss function for multi-organ segmentation and modify the segmentation mask labels to match the problem**

![Learning curve and accuracy - Task 2](/Lab5/results/task2/loss-accuracy.png)

The loss function used is "categorical_crossentropy". The validation accuracy and loss are:

|Accuracy  |Loss      |
|----------|----------|
|    0.8949|    0.1860|

![Example of segmentation - Task 2](/Lab5/results/task2/segmentation_example.png)

The predicted segmentation is again quite good even if not perferct. It succeeds in distinguishing the right from the left lung.

## Task 3

**With K-fold cross validation, you do not simply split the dataset into one training and one validation set, but rather test K different splits between training and validation sets. Train the model and report the performance metrics obtained**

We decided to integrate an auto-context into our U-net pipeline by adding one extra input channel.

The following metrics are calculated as the mean +/- deviation standard of the maximum metrics obtained from each fold training:

- Dice coeff = 0.008+-0.000
- Recall = 0.517+-0.077
- Precision = 0.996+-0.004

The task of segmentation with MRI images was quite challenging and we were not able to achieve good results as the graphs below show (one for each fold). The network seems unable to learn how to segment the images.

Fold 1 as validation (fold 2 + fold 3 as training):

![Result with fold 1 as validation - Task 3](/Lab5/results/task3/loss-accuracy_0.png)

Fold 2 as validation:

![Result with fold 2 as validation - Task 3](/Lab5/results/task3/loss-accuracy_1.png)

Fold 3 as validation:

![Result with fold 3 as validation - Task 3](/Lab5/results/task3/loss-accuracy_2.png)
