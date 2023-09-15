---
title: Report laboratory 2
author: Alessia Egano, Simone Bonino
---


# Report laboratory 2

## Task 5

### 5A

**Set the following parameters: # of epochs = 20; batch size = 8; number of feature maps at the first convolutional layer = 32 and learning rate = 0.00001 and then run the experiment. What are the value training and validation accuracies?**

|                    | Accuracy
|---------           |---------
|     Training       |0.519
|     Validation     |0.545

![Learning curve with epochs = 20, LR = 0.00001 and batch size = 8](/Lab2/images/task5/learning_curve20_00001.png)

**What can you infer from the learning curves?**

The network is not well trained and the loss value improves very little after each epoch.

![Accuracy curve with epochs = 20, LR = 0.00001 and batch size = 8](/Lab2/images/task5/accuracy_curve20_00001.png)

**Is it reasonable to make a decision based on this set-up of the parameters?**

No, the best accuracy of this set-up is pretty low and the predictions are close to be random. However, an higher number of epoch might improve this value since the curve is a monotonically increasing function.

### 5B

**Leave all the parameters from the previous task unchanged except for the n_epoch = 200. Compare the results with task 5A.**

![Learning curve with epochs = 200, LR = 0.00001 and batch size = 8](/Lab2/images/task5/learning_curve200_00001.png)

![Accuracy curve with epochs = 200, LR = 0.00001 and batch size = 8](/Lab2/images/task5/accuracy_curve200_00001.png)

As reported before, the performance improves with the increase of the number of epochs, with lower loss value and higher accuracy.

### 5C

**Keep all parameters from the last step except for the LR = 0.0001. Run the experiment with new LR and interpret the results. How do you evaluate the generalization power of the model? What are the values of training and validation accuracies?**

The loss value is lower than before because the model learns faster with a higher learning rate.

![Learning curve with epochs = 200, LR = 0.0001 and batch size = 8](/Lab2/images/task5/learning_curve200_0001.png)

The model performance with the validation set is similar to the training one, suggesting a good generalization power of the model (there's no evidence of overfitting).

![Accuracy curve with epochs = 200, LR = 0.0001 and batch size = 8](/Lab2/images/task5/accuracy_curve200_0001.png)

|                    | Accuracy
|---------           |---------
|     Training       |0.698
|     Validation     |0.730

### 5D

**What is the role of the first two convolutional layers?**

The main function of convolutional layers is to extract features from the input images.

### 5E

**What is the role of the last two dense layers?**

The role of dense layers is to classify images based on the output of the convolutional layer.

### 5F

**What is the major difference between the LeNet and MLP?**

LeNet is specifically designed for image processing using convolutional layers, while MLP is a feedforward neural network that can be used for classification and regression problems.

### 5G

**Look at the last layer of the network. How should we choose the number of neurons and the activation function of the last layer?**

The last layer of the network is related to the type of problem we are facing: in case of a binary classification the number of neuron should be 1, since the number of output neurons should be equal to the number of output associated with each input. The activation function should be a sigmoid which is able to return value closer to 0 or 1 (binary problem).

## Task 6

### 6A

**Read the skin images with the size of 128*128 and train the AlexNet model with the following parameter: batch size = 8; epochs = 50; n_base(Base) = 32; learning rate = 0.0001, and Adam as optimizer. Evaluate the model performance.**

The model performance is good when considering the training set but the same can't be said for the validation set: the model used on the validation set has an increase in the loss function throught the epochs and a lower accuracy value with respect to the training set one which reaches 1. This indicates the presence of overfitting of the model in the training phase. The same can be inferred by the learning curve which shows a much higher loss value for the validation set after the initial epochs.

![Learning curve with batch size = 8, epochs = 50, n_base(Base) = 32, learning rate = 0.0001 and Adam as optimizer](/Lab2/images/task6/learning_curve32_50_0001.png)

![Accuracy curve with batch size = 8, epochs = 50, n_base(Base) = 32, learning rate = 0.0001 and Adam as optimizer](/Lab2/images/task6/accuracy_curve32_50_0001.png)

### 6B

**Change the n_base parameter as 16 and 8 and run the model for 50 epochs. How do you interpret the observe results?**

*n_base = 16*

![Learing curve with batch size = 8, epochs = 50, n_base(Base) = 16, learning rate = 0.0001 and Adam as optimizer](/Lab2/images/task6/learning_curve16_50_0001.png)

![Accuracy curve with batch size = 8, epochs = 50, n_base(Base) = 16, learning rate = 0.0001 and Adam as optimizer](/Lab2/images/task6/accuracy_curve16_50_0001.png)

*n_base = 8*

![Learing curve with batch size = 8, epochs = 50, n_base(Base) = 8, learning rate = 0.0001 and Adam as optimizer](/Lab2/images/task6/learning_curve8_50_0001.png)

![Accuracy curve with batch size = 8, epochs = 50, n_base(Base) = 8, learning rate = 0.0001 and Adam as optimizer](/Lab2/images/task6/accuracy_curve8_50_0001.png)

What we can see from the learning curves of these two models is that by decreasing the number of kernel in the initial convolutional layer (n_base), the model improves slightly since they show less overfitting. Despite this, the model still shows overfitting and can't be considered well trained.

**Now, with n_base = 8, after each of the dense layer add a “drop out layer” with a drop out rate of 0.4 and train the model for 50 epochs. What is the effect of the drop out layer?**

The introduction of the drop out layers is able to resolve the overfitting problem as it can be seen in the accuracy plot: the accuracy curves of the training and validation sets are much more similiar than before suggesting that the model is able to predict correctly even new data set (different from training one). The maximum value of accuracy reached also confirms that the considered model has better performance than those without the drop out layer.

![Learning curve with “drop out layer”, batch size = 8, epochs = 50, n_base(Base) = 8, learning rate = 0.0001 and Adam as optimizer](/Lab2/images/task6/learning_curve8_50_0001_drop.png)

![Accuracy curve with “drop out layer”, batch size = 8, epochs = 50, n_base(Base) = 8, learning rate = 0.0001 and Adam as optimizer](/Lab2/images/task6/accuracy_curve8_50_0001_drop.png)

**Increase the number of epochs to 150. How do you explain the effect of increasing the number of epochs?**

The increase in the number of epochs leads to a worse model which is more similar in terms of accuracy and learning curve trends to the ones without the drop out layer, in which the overfitting is greater. 

The number of epochs is an important parameter that can lead to overfitting when it's set too high: the model becomes too speciliazed in predicting the training set and it's not able to predict well new data

![Learning curve with “drop out layer”, batch size = 8, epochs = 50, n_base(Base) = 8, learning rate = 0.0001 and Adam as optimizer](/Lab2/images/task6/learning_curve8_150_0001_drop.png)

![Accuracy curve with “drop out layer”, batch size = 8, epochs = 50, n_base(Base) = 8, learning rate = 0.0001 and Adam as optimizer](/Lab2/images/task6/accuracy_curve8_150_0001_drop.png)

### 6C

**Remove the drop out layers, set the parameters n_base=8, learning_rate=1e-5 and run the model for n_epochs =150, 350 epochs. How changing the learning rate parameter affect model performance?**

The change in the learning rate results in better models with higher accuracy, lower loss value and less overfitting. Notice that the model with the higher number of epochs for training still shows better results for the training set than the validation set, expecially for the loss value. We can assume by looking at the curves that the model with epochs = 150 is the better one since the curves for training and validation sets are much more similar, suggesting that higher numbers (epochs = 350) tends to overfit the model.

*epochs = 150*
![Learning curve with batch size = 8, epochs = 150, n_base(Base) = 8, learning rate = 1e-5 and Adam as optimizer](/Lab2/images/task6/learning_curve8_150_000001.png)

![Accuracy curve with batch size = 8, epochs = 150, n_base(Base) = 8, learning rate = 1e-5 and Adam as optimizer](/Lab2/images/task6/accuracy_curve8_150_000001.png)

*epochs = 350*
![Learning curve with batch size = 8, epochs = 350, n_base(Base) = 8, learning rate = 1e-5 and Adam as optimizer](/Lab2/images/task6/learning_curve8_350_000001.png)

![Accuracy curve with batch size = 8, epochs = 350, n_base(Base) = 8, learning rate = 1e-5 and Adam as optimizer](/Lab2/images/task6/accuracy_curve8_350_000001.png)

### 6D

**For the fix parameters of learning_rate = 1e-5, n_base=8, n_epochs = 150, change the batch size parameters as 2,4,8. Do you see any significant differences in model performance?**

By increasing the number of batch, the model becomes better at predicting the validation set and the efficiency of the algorithm improves.

*batch size = 2*
![Learning curve with batch size = 2, epochs = 350, n_base(Base) = 8, learning rate = 1e-5 and Adam as optimizer](/Lab2/images/task6/learning_curve8_150_000001_2.png)

![Accuracy curve with batch size = 2, epochs = 350, n_base(Base) = 8, learning rate = 1e-5 and Adam as optimizer](/Lab2/images/task6/accuracy_curve8_150_000001_2.png)

*batch size = 4*
![Learning curve with batch size = 4, epochs = 350, n_base(Base) = 8, learning rate = 1e-5 and Adam as optimizer](/Lab2/images/task6/learning_curve8_150_000001_4.png)

![Accuracy curve with batch size = 4, epochs = 350, n_base(Base) = 8, learning rate = 1e-5 and Adam as optimizer](/Lab2/images/task6/accuracy_curve8_150_000001_4.png)

*batch size = 8*
![Learning curve with batch size = 4, epochs = 350, n_base(Base) = 8, learning rate = 1e-5 and Adam as optimizer](/Lab2/images/task6/learning_curve8_150_000001.png)

![Accuracy curve with batch size = 4, epochs = 350, n_base(Base) = 8, learning rate = 1e-5 and Adam as optimizer](/Lab2/images/task6/accuracy_curve8_150_000001.png)

### 6E

**By finding the optimum values of batch_size, learning rate, and base parameters, train the model for 100 epochs and make sure it is not overfitted. Report the classification accuracy of the model.**

The optimal value of the parameters to avoid overfitting in the model are:
| Batch size  | Learning rate | Base
|---------    |---------      |---------
| 32          | 0.00001       | 8

The maximum accuracy reached for this model was 0.8550 (loss value = 0.4506)
The model does not show signs of overfitting in the accuracy curve, but the learning curve shows lower loss value for the training set suggesting the presence of some overfitting.

![Learning curve with batch size = 8, epochs = 100, n_base(Base) = 32, learning rate = 1e-5 and Adam as optimizer](/Lab2/images/task6/learning_curve_best.png)

![Accuracy curve with batch size = 8, epochs = 100, n_base(Base) = 32, learning rate = 1e-5 and Adam as optimizer](/Lab2/images/task6/accuracy_curve_best.png)

**Then, for this model, only change the optimizer algorithm from Adam to SGD and RMSprop and compare the observed results.**

The value of accuracy and loss obtained with Adam and RMSprop optimizer are very similar whereas the model obtained with SGD performs poorly and gives the same accuracy of a random classification.

|             | Accuracy    | Loss
|---------    |---------    |---------
| Adam        | 0.84        | 0.4520
| SGD         | 0.5         | 0.6920
| RMSprop     | 0.83        | 0.4524

![Learning curve with batch size = 8, epochs = 100, n_base(Base) = 32, learning rate = 1e-5 and Adam as optimizer](/Lab2/images/task6/learning_curve_best_Adam.png)

![Accuracy curve with batch size = 8, epochs = 100, n_base(Base) = 32, learning rate = 1e-5 and Adam as optimizer](/Lab2/images/task6/accuracy_curve_best_Adam.png)

### 6F

**“Binary cross entropy (BCE)” is not the only loss function for a binary classification task. Run the code again by changing the loss func into “hinge” with Adam optimizer. What is the major difference between the “BCE” and “hinge” in terms of model performance?**

The model performance for BCE is much better than the one with hinge. The model trained using the 'hinge' function has a maximum accuracy of 50% which decreases as the epochs increase. This suggest that Binary Cross Entropy is the best loss function to use to train the model.

|             | Accuracy    | Loss
|---------    |---------    |---------
| BCE         | 0.84        | 0.4520
| hinge       | 0.5         | 0.6967

![Learning curve with batch size = 8, epochs = 100, n_base(Base) = 32, learning rate = 1e-5, Adam as optimizer and hinge as loss function](/Lab2/images/task6/learning_curve_best_hinge.png)

![Accuracy curve with batch size = 8, epochs = 100, n_base(Base) = 32, learning rate = 1e-5, Adam as optimizer and hinge as loss function](/Lab2/images/task6/accuracy_curve_best_hinge.png)
