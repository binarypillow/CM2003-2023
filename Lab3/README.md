---
title: Report laboratory 2
author: Alessia Egano, Simone Bonino
---

# Report laboratory 3

⚠️ The complete results of this lab, such as saved training history, saved weights of the best model and graphs, can be found in the [results](/Lab3/results) folder.

## Task 1

### 1B)

**For the following fixed parameters, compile the model and find the optimum value of the learning rate that will lead to reliable classification performance over the validation set. What is the proper learning rate? n_epoch = 100, batch_s = 8, n_base = 8.**

*learning rate = 0.0001*
![Learing curve and accuracy - Task 1B](/Lab3/results/task1b/loss-accuracy_04.png)

*learning rate = 0.00001*
![Learing curve and accuracy - Task 1B](/Lab3/results/task1b/loss-accuracy_05.png)

*learning rate = 0.000001*
![Learing curve and accuracy - Task 1B](/Lab3/results/task1b/loss-accuracy_06.png)

The best model (minimum value of loss) is achieved with learning rate = 0.00001. With higher learning rate, the model is overfitted; whereas, with lower learning rate, the performance is worse

LR       | 0.0001 | 0.00001 | 0.000001 |
---------|--------|---------|----------|
Accuracy |    0.81|     0.85|      0.80|
Loss     |  0.4431|   0.4255|    0.6907|

### 1C)

**For the same setting as task 1B (only for n_epoch=150) compare the classification accuracy over the validation set between the AlexNet and VGG16 models. How do you justify the observed results?**

*Alexnet*
![Learing curve and accuracy AlexNet - Task 1C](/Lab3/results/task1c/loss-accuracy_A.png)

*VGG16*
![Learing curve and accuracy VGG16 - Task 1C](/Lab3/results/task1c/loss-accuracy_V.png)

Model    |AlexNet |   VGG16 |
---------|--------|---------|
Accuracy |   0.815|    0.845|

The classification accuracy is higher for the VGG16 model than AlexNet. This is expected since the VGG16 model has a deeper network of convolutional layer (8 more layers than AlexNet)

### 1D)

**Change the parameter n_base to 16. Train the model for 100 epochs with LR = 1e-5 and batch size of 8. Does increasing the number of feature maps affect model performance? Then, add a dropout layer with a rate of 0.2 for each of the dense layer. What was the effect of adding the dropout layer?**

*n_base=16*
![Learing curve and accuracy - Task 1D](/Lab3/results/task1d/loss-accuracy.png)

*n_base=16 with dropout layer*
![Learing curve and accuracy with dropout layer - Task 1D](/Lab3/results/task1d/loss-accuracy_D.png)

n_base   |       8|       16|16 with drop|
---------|--------|---------|------------|
Accuracy |   0.845|    0.840|       0.855|

The performance does not improve by increasing the parameter n_base as it can be seen by the slightly lower accuracy obtained.
On the contrary, by adding dropout layers after each dense layer, the model performs better but the improvement is small since there was no overfitting in the original model and the dropout layers help mainly in avoiding that behaviour

### 1E)

**So far, you classified the Skin cancer dataset with 3 different models named as LeNet, AlexNet as well VGG16. In general, what is the difference between these three models? Which of them yields more accurate classification results? Why? To evaluate the model performance, how do you assess the loss values? How can you prevent the model training from overfitting?**

The main difference between the models is the number of convolutional layers implemented: 2 for LeNet, 5 for AlexNet and 13 for VGG. This is the main reason why VGG has usually better performances with higher accuracy: the performance of the model improves by increasing the number of convolutional layers. The loss value used to evaluate model performance for CNN is usually Binary Cross Entropy, which can also be used in multiclass-classification since the calculated value is indipendent for each class.
The phenomena of overfitting happens when the model is fitting too much the training data after the training process and, for this reason, is not able to predict new data with the same accuracy. This will be represented in the loss plot by a low loss value close to 0 for the training set but an increasing loss for the validation set; whereas in the accuracy plot, the training accuracy will tend to be equal to 1 (perfect prediction) but the validation accuracy is going to be lower. The overfitting can happen when the dataset used for the training phase is not very big. To avoid it, we can increase the size of the dataset or implement dropout layers in architecture of the neural network, which randomly remove neurons from hidden layers and force the network to learn more robust feature that are less dependent on single neurons.

## Task 2

**Train the VGG16 model developed in Task1 with the following parameters to classify two types of bone fractures from the “Bone” dataset. Please note the size of the Bone images is quite large, so that it would take a longer time to read and load all the images.**

*n_base=8*
![Learing curve and accuracy - Task 1D](/Lab3/results/task2/loss-accuracy_8.png)

*n_base=8 with dropout layer*
![Learing curve and accuracy with dropout layer - Task 1D](/Lab3/results/task2/loss-accuracy_8_drop.png)

*n_base=16*
![Learing curve and accuracy - Task 1D](/Lab3/results/task2/loss-accuracy_16.png)

*n_base=16 with dropout layer*
![Learing curve and accuracy with dropout layer - Task 1D](/Lab3/results/task2/loss-accuracy_16_drop.png)

## Task 3

**With the implemented VGG models, which of the Skin/Bone image sets classified more accurately? Why? How do you make sure that achieved results are reliable?**

The VGG model gives better results for the classification of Bone fracture, with accuracy reaching 90% against the 85% of the Skin classification. The same can be seen from the loss curve. We assume that the performances obtained with Bone dataset are better because the AFF (Atypical Femur Fraction) and NFF (Normal Femur Fraction) are characterized by specific radiopgraphic features that can be detected by the network

|        |Accuracy  |Loss      |
|--------|----------|----------|
|    Skin|    0.8500|    0.4255|
|    Bone|    0.9071|    0.2415|

The table refers to VGG model with n_base = 8, batch_size = 8 and LR = 0.00001. The same consideration can be done for VGG models with different parameters.

## Task 4

**Modify the data loader to load the images along with their class labels properly. Extend the LeNet and AlexNet models for multi class classification tasks. Tune these two models by finding the optimum values of hyperparameters to get the best performance for each of the models and, then, compare the observed results between the two models. Report the learning curves for both of the loss and accuracy values (for train and test data).**

The optimal parameters found are the following:

| Parameters|    n_base| learning_rate| batch_size| dropout_rate|
|---------- |----------|--------------|-----------|-------------|
|     Values|         8|       0.00001|          8|          0.4|

The presence of the dropout layers help in improving the overfitting tendency and allows to obtain loss and accuracy curves that are closer to each other for training and validation set, especially with a higher value of dropout rate.

*AlexNet*
![Learing curve and accuracy AlexNet - Test1](/Lab3/results/task4/loss-accuracy_test5_A.png)

*LeNet*
![Learing curve and accuracy LeNet - Test1](/Lab3/results/task4/loss-accuracy_test5_L.png)

|        |Accuracy  |Loss      |
|--------|----------|----------|
| AlexNet|    0.9824|    0.0572|
|   LeNet|    0.9872|    0.0474|

The performances of LeNet and AlexNet are very similar in terms of accuracy and loss value. However it is important to notice that AlexNet has less tendency to overfit the dataset than LeNet, assuming the same values for the hyperparameters reported in the table above.

## Task 5

### 5A)

**Employ the AlexNet model with the following architecture: five convolutional blocks [with feature maps of the size of base, base*2, base*4, base*4, base*2 where base=8], followed by three dense layers with 64, 64, and 1 neuron, respectively. Add three max-pooling layers after 1st, 2nd, and 5th convolutional blocks. Set the following parameters: learning rate=0.0001, batch size=8, ‘relu’ as activation function, ‘Adam’ as optimizer, and image size=(128,128,1). Train this model for 50 epochs on skin images. What are the values of the train and validation accuracy? How do you interpret the learning curves?**

The results are the following:

*without dropout layers*
|             |Accuracy  |Loss      |
|-------------|----------|----------|
|     Training|    0.9309|    0.1748|
|   Validation|    0.8299|    0.4446|

![Learing curve and accuracy without dropout layer - Task 5A](/Lab3/results/task5a/loss-accuracy.png)

The overall result is good, but the trained model is overfitted and the validation set performs poorly compared to the training set.

**Add two drop out layers after the first two dense layers with the dropout rate of 0.4 and repeat the experiments and compare the results. What is the effect of adding drop out layers?**

*with dropout layers*
|             |Accuracy  |Loss      |
|-------------|----------|----------|
|     Training|    0.8399|    0.3774|
|   Validation|    0.8500|    0.4499|

![Learing curve and accuracy with dropout layer - Task 5A](/Lab3/results/task5a/loss-accuracy_D.png)

The dropout layers helps to reduce overfitting and increase the network's ability to generalise the unseen data of the validation set. This behaviour can be clearly seen by noticing that the train and validation curves are very similar to each other.

### 5B)

**With the same model and same settings, now insert a batch normalization layer at each convolutional block (right after convolution layer and before activation function). At which epoch do you observe the same training accuracy as task (a)? What is the value of final training accuracy? What is the effect of the batch normalization layer? Similar to task (a), do this task with and without drop out layers.**

The results are the shown below:

*without dropout layers*
|             |Accuracy  |Loss      |
|-------------|----------|----------|
|     Training|       1.0|    0.0041|

![Learing curve and accuracy without dropout layer - Task 5B](/Lab3/results/task5b/loss-accuracy.png)

The max training accuracy of task (a) (0.9309) is reached and exceeded at epoch 9.

The batch normalisation layers significantly accelerate the training process and the convergence during training is faster. For this reason, they help to stabilize the learning process.

*with dropout layers*
|             |Accuracy  |Loss      |
|-------------|----------|----------|
|     Training|    0.9490|    0.1174|

![Learing curve and accuracy with dropout layer - Task 5B](/Lab3/results/task5b/loss-accuracy_D.png)

The max training accuracy of task (a) with dropout (0.8399) is reached and exceeded at epoch 19.

As before, dropout levels produce a more generalisable model and reduce overfitting.

### 5C)

**Train again the same model with precisely the same parameters except learning rate = 0.00001 and epochs = 80 with and without batch normalization layers (in both cases, use the drop out layers). Focus on validation loss & accuracy. Which model resulted in higher validation accuracy? How do you explain the effect of batch normalization?**

*without normalization layers*
|             |Accuracy  |Loss      |
|-------------|----------|----------|
|   Validation|    0.8100|    0.5178|

![Learing curve and accuracy without normalization layer - Task 5C](/Lab3/results/task5c/loss-accuracy.png)

*with normalization layers*
|             |Accuracy  |Loss      |
|-------------|----------|----------|
|   Validation|    0.8149|    0.4521|

![Learing curve and accuracy with normalization layer - Task 5C](/Lab3/results/task5c/loss-accuracy_N.png)

The model resulted in slightly higher validation accuracy is the one with the normalization layers. As before, introducing normalization layers has stabilized the training process. However in this case, with a lower LR than task 5c) the convergence is reached more slowly.

### 5D)

**Keep the settings from the task1c unchanged and train the model for 150 epochs with and without batch normalization layers (in both cases, use the drop out layers). Which model yields more accurate results? Which of them has more generalization power?**

*without normalization layers*
|             |Accuracy  |Loss      |
|-------------|----------|----------|
|   Validation|    0.8299|    0.4736|

![Learing curve and accuracy normalization layer - Task 5D](/Lab3/results/task5d/loss-accuracy.png)

*with normalization layers*
|             |Accuracy  |Loss      |
|-------------|----------|----------|
|   Validation|    0.8399|    0.5679|

![Learing curve and accuracy with normalization layer - Task 5D](/Lab3/results/task5d/loss-accuracy_N.png)

The model produces more accurate results with the normalisation layers, but the one without normalisation seems to have greater generalising power, because the one with the normalisation layers tends to overfit after about 80 epochs. It is assumed that normalisation layers cannot avoid overfitted models when combined with a large number of epochs.

## Task 6

### 6A)

**Use the same model as task1d but set the “base” parameter as 32 and replace the batch normalization layers with spatial dropout layers at each convolutional blocks (after activation function, and before max-pooling). Set the dropout rate of spatial drop out layers as 0.1 and the rate of 0.4 for the normal drop out layers after the first two fully connected layers. Then let the model runs for 150 epochs with LR=0.00001. Save the loss and accuracy values for the validation data. Then, run the same model with the same settings but remove all the spatial drop out layers. Which of them converges faster? Why?**

*without spatial dropout layers*
|             |Accuracy  |Loss      |
|-------------|----------|----------|
|   Validation|    0.8249|    0.4755|

![Learing curve and accuracy without spatial dropout layer- Task 6A](/Lab3/results/task6a/loss-accuracy.png)

*with spatial dropout layers*
|             |Accuracy  |Loss      |
|-------------|----------|----------|
|   Validation|    0.8050|    0.5006|

![Learing curve and accuracy with spatial dropout layer - Task 6A](/Lab3/results/task6a/loss-accuracy_SD.png)

The one without spatial dropout layers converges faster because when the model uses dropout layers the memorization of training data is discouraged by dropping randomly feature maps.

### 6B)

**Repeat the task6a for 250 epochs with and without spatial dropout layers. In general, discuss how the drop out technique (spatial and normal one) would help the learning procedure.**

*without spatial dropout layers*
|             |Accuracy  |Loss      |
|-------------|----------|----------|
|   Validation|    0.8399|    0.4306|

![Learing curve and accuracy without spatial dropout layer - Task 6B](/Lab3/results/task6b/loss-accuracy.png)

*with spatial dropout layers*
|             |Accuracy  |Loss      |
|-------------|----------|----------|
|   Validation|    0.8450|    0.4725|

![Learing curve and accuracy with spatial dropout layer - Task 6B](/Lab3/results/task6b/loss-accuracy_SD.png)

Dropout is a regularisation technique used in neural networks to reduce the risk of overfitting and improve the generalisation capabilities of a model. By randomly deactivating a fraction of neurons (in one-dimensional space or in two-dimensional space) during each training iteration, this method prevents a single or multiple neuron from becoming too dominant.

## Task 7

**A practical way to perform the data augmentation technique is to develop a generator.**

An example of the results that can be achieved with the augmentation technique is shown in the image below.

![Sample image - Task 7](/Lab3/results/task7b/example.png)

A single image is rotated, flipped and scaled to increase the size of input data.

## Task 8

**Use the AlexNet model with the batch normalization layers, and drop out layers for the first two dense layers(rate=0.4). Set the “base” parameter as 64, and assign 128 neurons for the first dense layer and 64 for the second one. Set the optimizer=Adam, LR=0.00001, batch-size=8, and train the model on skin images for 80 epochs. How the data augmentation impact model training? Why?**

Data augmentation leads to improved performance because this method allows the network to be fed with more input data, generating new data by creating new versions of the original data. In machine learning, especially deep learning, a large amount of input data is crucial to achieve good results during training and, when only a small data set is available, augmentation is commonly used to artificially increase the input data.

![Learing curve and accuracy - Task 8](/Lab3/results/task8/loss-accuracy.png)

## Task 9

**Repeat task8 for VGG model for both of skin and bone data set.**

*skin set*
|             |Accuracy  |Loss      |
|-------------|----------|----------|
|   Validation|    0.8799|    0.3437|

![Learing curve and accuracy - Task 9](/Lab3/results/task9/loss-accuracy_skin.png)

*bone set*
|             |Accuracy  |Loss      |
|-------------|----------|----------|
|   Validation|    1.0   |    0.0007|

![Learing curve and accuracy - Task 9](/Lab3/results/task9/loss-accuracy_bone.png)

As seen in Task8, the increased data exposes the model to a wider range of variations in the input data and improves its performance on the validation data (reduction of overfitting). In both cases, the accuracy achieved is the highest to date.
