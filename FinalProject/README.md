---
title: Final Project
author: Alessia Egano, Simone Bonino
---

# Report Final Project

**Table of contents**

1. [Challenge](#challenge)
2. [Dataset](#dataset)
3. [Task](#task)
4. [Model used](#model-used)
5. [Tuning process](#tuning-process)
    1. [Dropout layer](#dropout-layer)
    2. [Dropout and normalization layer](#dropout-layer-and-batch-normalization)
6. [Results](#results)
    1. [Tuned model](#tuned-model)
    2. [Tuned model with random crops](#tuned-model-with-random-cropping)
    3. [Tuned model with random crops and transfer learning](#tuned-model-with-transfer-learning-vgg16)
7. [Conclusions](#conclusions)

# 1. Challenge

[RAVIR](https://ravir.grand-challenge.org/): A Dataset and Methodology for the Semantic Segmentation and Quantitative Analysis of Retinal Arteries and Veins in Infrared Reflectance Imaging.

## 2. Dataset

The images in RAVIR dataset were captured using infrared (815nm) Scanning Laser Ophthalmoscopy (SLO), which in addition to having higher quality and contrast, is less affected by opacities in optical media and pupil size. RAVIR images are sized at 768 × 768 and compressed in the Portable Network Graphics (PNG) format.

|*Train image* |*Train mask* |*Test image* |
|--------------|-------------|-------------|
|23 images |23 masks |19 images |
|![Example of training image](/FinalProject/data/train/training_images/IR_Case_011.png) | ![Example of training mask](/FinalProject/data/train/training_masks/IR_Case_011.png) | ![Example of testing image](/FinalProject/data/test/IR_Case_006.png) |

## 3. Task

The objective assessment of retinal vessels has long been considered a surrogate biomarker for systemic vascular diseases, and with recent advancements in retinal imaging and computer vision technologies, this topic has become the subject of renewed attention.

The aim of this challenge is to segment the vessel type using a deep learning-based model without the use of extensive post-processing. The model needs to be trained using the images and masks contained in the *train* folder and tested on images contained in the *test* folder.

* Predictions obtained in the texting phase should be a PNG file as 2D maps containing artery and vein classes with size (768,768).
* Artery and vein classes should have labels of 128 and 256 respectively. Background should have a label of 0.
* The filenames must exactly match the name of the images in the test set [IR_Case_006.png, ..., IR_Case_060.png].
* Predictions in the correct format should be placed in a folder named test and submit a zipped version of this folder to the server.

The Dice and Jaccard scores will be calculated for every image in the test set for both artery and vein classes. The leaderbord is sorted on the basis of the best average Dice score.

## 4. Model used

The U-Net model is a convolutional neural network that consists of a contracting path (left side) and an expansive path (right side). The *contracting path* follows the typical architecture of a convolutional network. It consists of the repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling. At each downsampling step we double the number of feature channels. Every step in the *expansive path* consists of an upsampling of the feature map followed by a 2x2 convolution (“up-convolution”) that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU. The cropping is necessary due to the loss of border pixels in every convolution. At the final layer a 1x1 convolution is used to map each feature vector to the desired number of classes. In total the network has 23 convolutional layers.

## 5. Tuning process

Parameters were firstly tuned by compiling and fitting the model using 80% of the original training set for the training process and 20% for the validation process but a high variability in the performance metrics was observed in each run. This high variability was assumed related to the different quality of images selected randomly to create the training set.

The **5-fold cross-validation** was implemented to avoid this outcome: the parameters with the higher Mean Dice Score over the 5 folds was selected as optimal.

Note that for every model training run, a number of **epochs = 300** was used, after noticing that the performance metrics usually stabilize around 200 epochs.

The results of the hyperparameters fine tuning for the simple U-net are reported below:

|Base   |Learning rate |Batch size |Dice Score     |Dice Arteries   |Dice Veins     |Jaccard        |Jaccard Arteries|Jaccard Veins |
|-------|--------------|-----------|---------------|----------------|---------------|---------------|---------------|---------------|
|      8|         0.001|          2|0.5389 ± 0.0840|0.4866 ± 0.0844|0.6143 ± 0.0717|0.3834 ± 0.0852|0.3351 ± 0.0818|0.4528 ± 0.0790|
|      8|         0.001|          4|0.4826 ± 0.1221|0.3520 ± 0.2153|0.6258 ± 0.0722|0.3488 ± 0.1076|0.2417 ± 0.1721|0.4658 ± 0.0850|
|      8|        0.0001|          2|0.4560 ± 0.0752|0.3473 ± 0.1306|0.5733 ± 0.0493|0.3124 ± 0.0617|0.2247 ± 0.0952|0.4061 ± 0.0500|
|      8|        0.0001|          4|0.4345 ± 0.1063|0.2982 ± 0.1594|0.5818 ± 0.0532|0.2964 ± 0.0811|0.1880 ± 0.1098|0.4139 ± 0.0538|
|     16|         0.001|          2|0.3778 ± 0.1334|0.1870 ± 0.1708|0.5813 ± 0.1825|0.2701 ± 0.1063|0.1149 ± 0.1227|0.4343 ± 0.1632|
|     16|         0.001|          4|0.5961 ± 0.0916|0.5466 ± 0.1041|0.6580 ± 0.0720|0.4403 ± 0.1009|0.3944 ± 0.1097|0.4991 ± 0.0858|
|     16|        0.0001|          2|0.4852 ± 0.1093|0.3921 ± 0.1600|0.5911 ± 0.0632|0.3400 ± 0.0940|0.2654 ± 0.1257|0.4255 ± 0.0653|
|     16|        0.0001|          4|0.4482 ± 0.1097|0.3104 ± 0.1870|0.5997 ± 0.0740|0.3157 ± 0.0944|0.2060 ± 0.1467|0.4361 ± 0.0797|

Learning curve, Dice score and Jaccard score are reported below for the optimal parameters selected.

* *Model with base = 8, batch size = 2*
![Learning curve and accuracy - simple model](/FinalProject/results/tuning/model_8_001_2/loss-dice-jaccard_CV.png)

* *Model with base = 16, batch size = 4*
![Learning curve and accuracy - simple model](/FinalProject/results/tuning/model_16_001_4/loss-dice-jaccard_CV.png)

### 5.1 Dropout layer

Dropout layers were introduced to reduce the tendency of the model to overfit since the performance of the training set was better than the validation one. Fine parameter tuning was applied to identify the optimal value of dropout rate, considering the optimal parameters value reported above.

|Base  |Learning rate |Batch size |Dropout rate  |Dice Score     |Dice Arteries  |Dice Veins     |Jaccard        |Jaccard Arteries|Jaccard Veins  |
|------|--------------|-----------|--------------|---------------|---------------|---------------|---------------|----------------|---------------|
|     8|         0.001|          2|           0.2|0.5431 ± 0.0808|0.4846 ± 0.0784|0.6241 ± 0.0666|0.3880 ± 0.0830| 0.3338 ± 0.0780|0.4620 ± 0.0747|
|    16|         0.001|          4|           0.2|0.4999 ± 0.1135|0.3497 ± 0.2171|0.6596 ± 0.0794|0.3664 ± 0.1015| 0.2377 ± 0.1728|0.5017 ± 0.0937|
|     8|         0.001|          2|           0.4|0.4394 ± 0.1467|0.2891 ± 0.2259|0.5992 ± 0.0758|0.3142 ± 0.1238| 0.1980 ± 0.1754|0.4367 ± 0.0804|
|    16|         0.001|          4|           0.4|0.5117 ± 0.1544|0.3870 ± 0.3419|0.6473 ± 0.0832|0.3783 ± 0.1401| 0.2760 ± 0.1977|0.4887 ± 0.0961|

The learning curve, dice score and jaccard score curve for the 5 folds of the best model are reported in the figure below.

*Model with base = 8, batch size = 2, dropout rate = 0.2*
![Learning curve and accuracy - model with dropout layer](/FinalProject/results/tuning/model_8_001_2_drop02/loss-dice-jaccard_CV.png)

### 5.2 Dropout layer and batch normalization

As a final step of the tuning process, batch normalization was introduced to see which model performed better. Batch normalization is a method used to make training of artificial neural networks faster and more stable through normalization of the layers' inputs by re-centering and re-scaling. The results are reported below.

|Base  |Learning rate |Batch size |Dropout rate  |Dice Score     |Dice Arteries  |Dice Veins     |Jaccard        |Jaccard Arteries|Jaccard Veins  |
|------|--------------|-----------|--------------|---------------|---------------|---------------|---------------|----------------|---------------|
|     8|         0.001|          2|           0.2|0.6255 ± 0.0589|0.5944 ± 0.0665|0.6760 ± 0.0437|0.4645 ± 0.0669| 0.4321 ± 0.0729|0.5156 ± 0.0537|
|    16|         0.001|          4|           0.2|0.6361 ± 0.0841|0.6051 ± 0.0921|0.6811 ± 0.0663|0.4788 ± 0.0961| 0.4469 ± 0.1018|0.5242 ± 0.0813|
|     8|         0.001|          2|           0.4|0.6266 ± 0.0588|0.5930 ± 0.0694|0.6703 ± 0.0405|0.4640 ± 0.0648| 0.4298 ± 0.0738|0.5083 ± 0.0487|
|    16|         0.001|          4|           0.4|0.6262 ± 0.0839|0.5918 ± 0.0975|0.6717 ± 0.0624|0.4681 ± 0.0946| 0.4337 ± 0.1058|0.5133 ± 0.0765|

The learning curve, dice score and jaccard score curve for the 5 folds of the best model are reported in the figure below.

*Model with base = 16, batch size = 4, dropout rate = 0.2*
![Learning curve and accuracy - model with dropout layer and batch normalization](/FinalProject/results/tuning/model_16_001_4_drop02_norm/loss-dice-jaccard_CV.png)

## 6. Results

### 6.1 Tuned model

The best model found with fine tuning is the U-Net with dropout layer and batch normalization with the following parameters.

|Base   |Learning rate |Batch size |Dropout rate|Normalization |
|-------|--------------|-----------|------------|--------------|
|     16|         0.001|          4|         0.2|          True|

*U-Net with dropout layer, batch normalization and optimal parameter values*
|Dice Score      |Dice Arteries   |Dice Veins      |Jaccard         |Jaccard Arteries|Jaccard Veins   |
|----------------|----------------|----------------|----------------|----------------|----------------|
| 0.8847 ± 0.1189| 0.8815 ± 0.1306| 0.8880 ± 0.1082| 0.8110 ± 0.1639| 0.8088 ± 0.1775| 0.8133 ± 0.1516|

![Learning curve and accuracy - model with dropout layer and batch normalization](/FinalProject/results/testing/simple/loss-dice-jaccard.png)

Example of predicted segmentation compared to original image and true mask are shown below:

![Example of segmentation - model with dropout layer and batch normalization](/FinalProject/results/testing/simple/segmentation/IR_Case_011.png)

![Example of segmentation - model with dropout layer and batch normalization](/FinalProject/results/testing/simple/segmentation/IR_Case_021.png)

The segmentation predictions of the test set images were submitted and the result was the following:

![Leaderboard result - model with dropout layer and batch normalization](/FinalProject/submissions/simple/result_simple.png)

|![Example of test set image](/FinalProject/data/test/IR_Case_006.png) | ![Example of test set prediction - model with dropout layer and batch normalization](/FinalProject/results/testing/simple/prediction/test/IR_Case_006.png)|
|---|---|

Note that in the mask the *white* area represents the veins and the *gray* area represents the arteries.

### 6.2 Tuned model with random cropping

The next step to improve the model prediction was to implement random cropping of the training set to increase the number of samples and reduce the tendency of overfitting by generalizing the model with this type of data augmentation.
For each image in the training set, we applied the defined function *randomCrop()* (can be found in [DataLoader.py](/FinalProject/utils/DataLoader.py)) which randomly crops each image to a **img_w_crop** x **img_h_crop** dimension for **n_crops** number of times. The cropped images were brought back to the original dimension by applying *zero padding* to be able to train the model.

Fine tuning has been applied to select the optimal values of parameters **img_w_crop**,**img_h_crop** dimension of the cropped image and **n_crops** number of crops.

* Tuning of dimension of crops with fixed number of crops (n_crops = 4)
* Tuning of number of crops with optimal dimension of crops (img_w_crop = img_h_crop = 4)

The results are reported below:

|Number of crops|Dimension of crops|Dice Score     |Dice Arteries  |Dice Veins     |Jaccard        |Jaccard Arteries|Jaccard Veins |
|---------------|------------------|---------------|---------------|---------------|---------------|----------------|---------------|
|              4|               256|0.6735 ± 0.0469|0.6306 ± 0.0631|0.7163 ± 0.0473|0.5119 ± 0.0523|0.4636 ± 0.0671|0.5601 ± 0.0570|
|              4|               384|0.7566 ± 0.0489|0.7378 ± 0.0591|0.7753 ± 0.0452|0.6116 ± 0.0630|0.5881 ± 0.0744|0.6352 ± 0.0598|
|              4|               512|0.8504 ± 0.0216|0.8419 ± 0.0251|0.8590 ± 0.0231|0.7406 ± 0.0326|0.7277 ± 0.0374|0.7535 ± 0.0355|
|              4|               640|0.9061 ± 0.0231|0.9059 ± 0.0258|0.9062 ± 0.0227|0.8292 ± 0.0381|0.8291 ± 0.0423|0.8293 ± 0.0378|
|              6|               640|0.9298 ± 0.0176|0.9291 ± 0.0208|0.9305 ± 0.0159|0.8694 ± 0.0305|0.8683 ± 0.0359|0.8705 ± 0.0277|
|              8|               640|0.9383 ± 0.0205|0.9391 ± 0.0229|0.9376 ± 0.0207|0.8846 ± 0.0358|0.8860 ± 0.0399|0.8832 ± 0.0362|
|             10|               640|0.9562 ± 0.0208|0.9554 ± 0.0226|0.9570 ± 0.0207|0.9169 ± 0.0367|0.9154 ± 0.0399|0.9183 ± 0.0366|

The learning curve, dice score and jaccard score curve for the best model are reported in the figure below.

*Number of crops = 10, dimention of crops = 640*
![Learning curve and accuracy - model with dropout layer and batch normalization + 10 640x640 crops](/FinalProject/results/testing/10crop640x640/loss-dice-jaccard.png)

Example of predicted segmentation compared to original image and true mask are shown below:

![Example of segmentation - model with dropout layer and batch normalization + 10 640x640 crops](/FinalProject/results/testing/10crop640x640/segmentation/IR_Case_011_2.png)

![Example of segmentation - model with dropout layer and batch normalization + 10 640x640 crops](/FinalProject/results/testing/10crop640x640/segmentation/IR_Case_021_2.png)

The segmentation predictions of the test set images were submitted and the result was the following (a bit better than the previous attempt):

![Leaderboard result - model with dropout layer and batch normalization + 10 640x640 crops](/FinalProject/submissions/10crop640x640/result_10crop640x640.png)

|![Example of test set image](/FinalProject/data/test/IR_Case_006.png)|![Example of test set prediction - model with dropout layer and batch normalization + 10 640x640 crops](/FinalProject/results/testing/10crop640x640/prediction/test/IR_Case_006.png)|
|---|---|

### 6.3 Tuned model with random crops and transfer learning (VGG16)

To further improve the performance of the model in the segmentation task, transfer learning was implemented in the model training process. **Transfer learning** in a CNN refers to using a pre-trained model on a similar task as a starting point for training a new model on a different task. In this case the pre-trained model used was a *VGG16* model. This was implemented by modifying the original U-Net model architecture by introducing already trained VGG16 layers in the encoing path and bottleneck, but keeping the same decoding path as in the original network.

For this last step, the model was trained using the optimal parameters value found and reported in the paragraph [6.2](#tuned-model-with-random-cropping): n_crops = 10, img_w_crops = 640.

Note that for this model training **batch_size = 2** was used even though the optimal value found in the fine tuning process was 4, due to limitation of the GPU used to run the code (not enough memory to allocate).

|Number of crops|Dimension of crops|Dice Score     |Dice Arteries  |Dice Veins     |Jaccard        |Jaccard Arteries|Jaccard Veins  |
|---------------|------------------|---------------|---------------|---------------|---------------|----------------|---------------|
|              10|               640|0.9562 ± 0.0208|0.9554 ± 0.0226|0.9570 ± 0.0207|0.9169 ± 0.0367|0.9154 ± 0.0399|0.9183 ± 0.0366|

The learning curve, dice score and jaccard score curve for the best model are reported in the figure below.

*Number of crops = 10, dimention of crops = 640 with transfer learning*
![Learning curve and accuracy - model with dropout layer and batch normalization + 10 640x640 crops + transfer](/FinalProject/results/testing/10crop640x640_tra/loss-dice-jaccard.png)

Example of predicted segmentation compared to original image and true mask are shown below:

![Example of segmentation - model with dropout layer and batch normalization + 10 640x640 crops + transfer](/FinalProject/results/testing/10crop640x640_tra/segmentation/IR_Case_011_2.png)

![Example of segmentation - model with dropout layer and batch normalization + 10 640x640 crops + transfer](/FinalProject/results/testing/10crop640x640_tra/segmentation/IR_Case_021_2.png)

The segmentation predictions of the test set images were submitted and the result was the following (the best result we reached):

![Leaderboard result - model with dropout layer and batch normalization + 10 640x640 crops + transfer](/FinalProject/submissions/10crop640x640_tra/result_10crop640x640_tra.png)

|![Example of test set image](/FinalProject/data/test/IR_Case_006.png)|![Example of test set prediction - model with dropout layer and batch normalization + 10 640x640 crops  + transfer](/FinalProject/results/testing/10crop640x640_tra/prediction/test/IR_Case_006.png)|
|---|---|

## 7. Conclusions

The best performance was obtained by applying to the U-Net model the transfer learning and the random cropping techniques to the model with the best tuned parameters. The optimal model parameter values are reported in the table below:

|Base   |Learning rate |Batch size |Dropout rate|Normalization |Number of crops|Dimension of crops|
|-------|--------------|-----------|------------|--------------|---------------|------------------|
|     16|         0.001|          4|         0.2|          True|             10|               640|

The model was classified in the **70th** place of the leaderboard, reaching an overall good result. Considering that this challenge represents our first experience with deep learning, we are satisfied with what we achieved.

---
⚠️ All the graphs produced working on this project can be found in the [results](/FinalProject/results) folder.
