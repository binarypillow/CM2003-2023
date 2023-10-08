---
title: Report laboratory 4
author: Alessia Egano, Simone Bonino
---

# Report laboratory 4

⚠️ The complete results of this lab, such as saved training history, saved weights of the best model and graphs, can be found in the [results](/Lab4/results) folder.

## Task 2

**Train the fine-tuned model with skin and bone images. Compare the observed results from transfer learning against the ones you already achieved by training the models from scratch. Describe how transfer learning would be useful for training. How can you make sure that the results are reliable?**

*Skin*
![Learing curve and accuracy - Task 9](/Lab3/results/task9/loss-accuracy_skin.png)
*Skin - Transfer learning*
![Learing curve and accuracy for Skin - Task 2](/Lab4/results/task2/loss-accuracy_skin.png)
*Bone*
![Learing curve and accuracy - Task 9](/Lab3/results/task9/loss-accuracy_bone.png)
*Bone - Transfer learning*
![Learing curve and accuracy for Bone - Task 2](/Lab4/results/task2/loss-accuracy_bone.png)

|        |                 |Accuracy  |Loss      |
|--------|-----------------|-----------|----------|
|    Skin|               - |     0.8800|    0.3245|
|        |Transfer learning|     0.8799|    0.3437|
|    Bone|               - |       1.0*|    0.2360|
|        |Transfer learning|        1.0|    0.0007|

*overfitting

The results show that the models trained using transfer learning are able to reach higher accuracy with a lower number of epochs. It is also important to note that the trasnfer learning method is prone to overfitting.

Transfer learning is a machine learning technique that leverages pre-trained models to improve the performance of a model on a specific task or domain. It is especially useful in scenarios where you have limited data or computational resources because it allows you to take advantage of knowledge learned from a related task or dataset.

To test of the results are reliable we should apply the model for prediction on a new test set of never seen before samples. Furthermore, the visualization of activation maps can be useful in understanding the features identified by the neural network for the classification task.

## Task 3

**Design a VGG16 model. Train the model with data augmentation techniques with parameters as : base=8; batch_s = 8; LR=1e-5; img_size = 128*128, epoch = 80. After the training process, follow the implementation below and interpret the observed results. What can you infer from visualization of the class activation maps?**

![Learing curve and accuracy - Task 3](/Lab4/results/task3/loss-accuracy.png)
*Original sample image*
![Original image - Task 3](/Lab4/results/task3/activation_maps.png)
*Activation map of sample image*
![Superimposed activation map - Task 3](/Lab4/results/task3/superimposed_activation_maps.png)

|Accuracy  |Loss      |
|----------|----------|
|    0.9982|    0.0093|

The model trained with data augmentation shows very good results in terms of accurcay and loss function
The activation map shows that the region of the image that was mostly used for the classification are the borders.
The data augmentation technique can improve the generalization capability of the model is to help the model learn more robust and invariant features from the training data, which can lead to better generalization on unseen or real-world data.
Based on the activation maps pbtained, we can assume that the classification is not based on visual features regarding the different types fractures but on writings on the image in which the right type of fracture (class) is reported. For this reason, the good results obtained in terms of perforamance are not reliable.

## Bonus Task

**Design a N-Layer Residual network for the classification task. Try to find the optimum hyperparameters such as batch size, depth of model, learning rate etc. Train your model with data augmentation for X-ray images. Apply the followings in your implementation:**

Note that the results reported below are refred to a model that has been trained **without** data augmentation

*Optimal hyperparameters*

| parameter| bacth size| depth| learning rate|
|----------|-----------|------|--------------|
|     value|          8|    10|         0.001|

### B2)

**Split your data through a 3-fold cross-validation approach and report the classification accuracy of your model as mean∓variation scores over the validation sets.**

Accuracy = 0.977 ∓ 0.008

## Task 4

### 4A)

**Train the U-Net model with the given parameters and report the final results.**

![Learing curve and accuracy - Task 4A](/Lab4/results/task4/loss-accuracy_a.png)

|Accuracy  |Loss      |
|----------|----------|
|    0.9602|    0.0545|

The simple application of U-Net model without any hyperparmeters tuning tecniques gives good results in terms of accuracy and loss

![Example of segmentation - Task 4A](/Lab4/results/task4/segmentation_example_a.png)

### 4B)

**Keep all the parameters from the previous exercise the same, only replace the BCE loss function with “Dice loss” and repeat the exercise. Is there any difference between model performance over validation set when you changed the loss functions? Do you expect to observe similar results when you deal with more challenging segmentation tasks? Dealing with cases in which the size of the target to be segmented would be much smaller than the image (imbalance between the background and foreground). Which loss function would you choose? Discuss it.**

![Learing curve and accuracy - Task 4B](/Lab4/results/task4/loss-accuracy_b.png)

|Accuracy  |Loss      |
|----------|----------|
|    0.9673|   -0.9673|

The results obtained with Dice loss function are better both in terms of accuracy and loss compared to the model trained with Binary Cross Entropy.
In cases in which the size of the target to be segmented is much smaller than the image, using Weighted Dice Loss as loss function can help emphasize the accurate segmentation of these critical structures. In general, Weighted Dice Loss can be particularly useful when there is a significant class imbalance or when certain classes are more critical than others.

![Example of segmentation - Task 4B](/Lab4/results/task4/segmentation_example_b.png)

### 4C)

**Repeat the tasks 4a and 4b by including the dropout layer (rate=0.2). Does it affect model performance? What about the learning curves?**

*Binary Cross Entropy*
![Learing curve and accuracy - Task 4C_a](/Lab4/results/task4/loss-accuracy_c_a.png)

*Dice loss*
![Learing curve and accuracy - Task 4C_b](/Lab4/results/task4/loss-accuracy_c_b.png)

|     |Accuracy  |Loss      |
|-----|----------|----------|
|  BCE|    0.9573|    0.0442|
| Dice|    0.9686|   -0.9686|

The model seems to perform better without the dropout layer even though the difference in the performance metrics obtained is very small. This is in accordance with the fact that the dropout out layers are mainly used to reduce overfitting, which is not present in our case, and not for improving performance.
Furthermore, by looking at the learning curve, we can see that the model with dropout layers reaches values around the minimum loss slower than the one without, at around 20 epochs.

*Binary Cross Entropy*
![Example of segmentation - Task 4C_a](/Lab4/results/task4/segmentation_example_c_a.png)

*Dice loss*
![Example of segmentation - Task 4C_b](/Lab4/results/task4/segmentation_example_c_b.png)

### 4D)

**Increase the model capacity by setting base=32. Repeat tasks 4c and evaluate the results.**

![Learing curve and accuracy - Task 4D](/Lab4/results/task4/loss-accuracy_d.png)

|Accuracy  |Loss      |
|----------|----------|
|    0.9742|   -0.9742|

The result show an improvement in the performance with higher values of Dice coefficient and loss, in accordance with the increase of base number of neurons in each convolutional layer and the consequential increase in complexity of the whole neural network.

![Example of segmentation - Task 4D](/Lab4/results/task4/segmentation_example_d.png)

### 4E)

**Repeat the task 1c with setting the base=16, batch norm=True, and train the model by applying augmentation technique on both images and masks: rotation range=10; width and height shift ranges=0.1; zoom range = 0.2, and horizontal flip. Could image augmentation improve the generalization power of the model?**

![Learing curve and accuracy - Task 4E](/Lab4/results/task4/loss-accuracy_e.png)

|Accuracy  |Loss      |
|----------|----------|
|    0.7893|   -0.7893|

The results obtained show a drastic decrease in the Dice coefficient and a worsening of the performance of the model.
Generally speaking, the data augmentation technique can improve the generalization capability of the model because it helps the model to learn more robust and invariant features from the training data, which can lead to better generalization of unseen or real-world data.
In this case, the application of the augmentation method gives as a result an approssimated segmentation, as it can be seen by the figure reported below.

![Example of segmentation - Task 4E](/Lab4/results/task4/segmentation_example_e.png)
