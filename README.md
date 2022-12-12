# Pytorch-landscape-classification

## Task:
Using a neural network to classify landscapes with Pytorch <br>

## Description: <br>
During this project I developed a model which can distinguish between different landscapes. It gets an image as an input and then outputs one of 6 classes. 

The data can be found here: <br>
https://www.kaggle.com/datasets/puneet6060/intel-image-classification<br>

## Details: 

The first architecture I have used was a regular CNN which failed the converge to a reasonable accuracy. I therfore used this opportunity to get my hands on one of the more advanced architectures, the "ResNet" which improved the accuracy of the model quite a lot. The current version of the model has 4 residual blocks which are implemented in a way that even after rescaling the output dimensions, the previous layer to be added and the output still form an identity. More details and the implementation can be found in the src folder.  <br>

Regular training did not suffice in order to reach a good accuracy so I have used methods like gradient clipping and using a [cyclic learning rate](https://arxiv.org/abs/1506.01186). <br>
The model eventually reached a 95% training accuracy and a 75% testing accuracy on a resnet built and trained by myself. <br>
As a comparison I have finetuned ResNet50 with its original weights over 20~ epochs which reaches a 85% accuracy. It has quickly reached that accuracy since the model was able to generalise based on its pretrained weights. <br>

My first thought was that the low accuracy might be due to overfitting but after further data augmentation the model still failed to generalize. </b>
Upon observing the test set further I have found out that many images have been assigned wrong labels which lead to wrong classifications. </br>

### Sidenotes:
I have impmented some helper and utility functions in scripts which can be found in the src folder. They serve the purpose of keeping the notebook as clean as possible and for validation. The notebook describes how to use these functions. 
The website folder contains the django app I created to use the model.

## Next steps:
Work on generalization and overfitting.
More validation methods.
