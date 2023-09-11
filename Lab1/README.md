# Report laboratory 1

---
title: Report laboratory 1
author: Alessia Egano, Simone Bonino
---

## Task 1

**Run the above code and interpret the results. Please note the output of the model is the prediction of class labels of each of the four points. If you run the code several times, will you observe the same results?**

No, the results change at each iteration.

**Why?**

The weights are initialized randomly with np.random.rand(), so the starting point is different at each run.
Keep the parameter “n_unit=1” and increase the number of iterations starting from 10, 50, 100, 500, 2000, and compare the loss values. What can you conclude from increasing the number of iterations?
The predictions improve (the loss values decrease) as the number of iterations increases (number of times the algorithm's parameters are updated).
| Iterations | Lowest SSD
|---------   |-----------
|10          |0.81
|50          |0.63
|100         |0.33
|500         |0.26
|2000        |0.25

**Now, with a fixed value of “iterations =1000”, increase the parameter “n_unit” to 2, 5, 10 and interpret the results.**

The predictions improve (the loss values decrease) as the number of neurons increases. The lowest SSD is reached with iterations = 1000 and n_unit = 10; however, the degree of improvement is small after 5 neurons as it can be seen by comparing the difference between the last two lowest SSD values. This consideration confirms the rule of thumb: “the number of hidden neurons should be between the size of the input layer and the size of the output layer”.
| Neurons | Lowest SSD
|---------|-----------
|1        |0.255463
|2        |0.001603
|5        |0.000998
|10       |0.000846

## Task 2

**Repeat task1 for XOR logic operator. For fixed values of parameters (iterations=2000, and n_unit=1), which of the AND or XOR operators has lower loss values?**

AND has lower loss values than XOR; In fact, the lowest XOR SSD (0.949776) with 2000 iterations which is higher even than the lowest XOR SSD (0.81) with only 10 iterations (for an equal number of units).

**Why?**

The neural network is better at predicting the AND outputs since they are linearly separable in the 2D plane.

**Increase the number of neurons in the hidden layer (n_unit) to 2, 5, 10, 50. Does increasing the number of neurons improve the results?**

The predictions improve (the loss values decrease) as the number of neurons increases.
The lowest SSD is reached with iterations = 2000 and n_unit = 10.

| Neurons | Lowest SSD
|---------|-----------
|1        |0.949776
|2        |0.502185
|5        |0.001319
|10       |0.001252
|50       |1.999999

**Why?**

As in task 1, the performance is equally good both for n_unit = 5 and 10 but, according to the rule of thumb, the SSD increases drastically if the number of neurons is too high compared to the size of input and output layers.

## Task 3

**In the above code, change the parameter “n_unit” as 1, 10 and interpret the observed results.**

The results show improvement in the prediction when n_unit = 10 is used. Moreover, with n_unit = 10 the predicted outputs at different runs are similar to each other, while with n_unit = 1 the values are not very consistent.

| Neurons | Lowest SSD
|---------|------------
| 1       | 0.872
|10       | 0.584

## Task 4

**How do you interpret the observed values of loss and accuracy values? Is the number of epochs enough to make a good decision about model performance?**

The best model is reached at the end of the model fitting process after 50 epochs with loss = 0.67 and accuracy =
Based on the trend of the learning curve, we can conclude that the number of epochs is not enough to make a decision about the model since the loss value is doesn't stabilize at the end but keeps decreasing, suggesting the possibility of an even better model with lower loss.

![Learning curve with epochs = 50 and LR = 0.0001](/Lab1/images/learning_curve50_0001.png)

**For the same number of epochs, reduce the learning rate parameter to 0.1 and interpret the results.**

By increasing the learning rate parameter, the learning curve stabilizes around loss = 0.7 after the first few epochs. We can assume that this learning rate parameter value is too high to perform a good selection of the best model.

![Learning curve with epochs = 150 and LR = 0.1](/Lab1/images/learning_curve50_1.png)

**Now increase the number of epochs to 150 with LR=0.0001. Does this model have enough capacity to yield acceptable results?**

The best model selected using these value of epochs and LR can be considered a good model given the trend of the learning curve that tend to stabilize itself around loss = 0.64

![Learning curve with epochs = 150 and LR = 0.0001](/Lab1/images/learning_curve150_0001.png)

**Increase the “base_dense” parameter to 256 and compare the results with the case of “base_dense=64”. Is increasing the model capacity helpful to improve the model performance? Why?**

The shape of the learning curve is very similar in the 2 cases considered but with an increase in the initial number of neurons in the network we obtain a lower loss and therefore a better model and performance.

![Learning curve with epochs = 150, LR = 0.0001 and 256 neurons](/Lab1/images/learning_curve150_0001_256.png)

The table below reports the val loss of the best model selected as the one with the lowest val loss in the learning curve. The val accuracy is not the maximum value, but the one corrisponding to the epoch with the lowest val loss (best model).

| Epochs | Learning rate | Base dense | Val loss | Val acc |
|--------|---------------|------------|----------|---------|
| 50     | 0.0001        | 64         | 0.6824   | 0.6400  |
| 50     | 0.1           | 64         | 0.6931   | 0.5000  |
| 150    | 0.0001        | 64         | 0.6169   | 0.7300  |
| 150    | 0.0001        | 256        | 0.6190   | 0.6850  |
