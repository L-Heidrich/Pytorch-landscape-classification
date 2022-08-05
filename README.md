# Pytorch-landscape-classification

## Task:
Using a recurrent neural network to classify landscapes with Pytorch <br>

## Description: <br>
During this project I have used methods like gradient clipping and using a [cyclic learning rate](https://arxiv.org/abs/1506.01186). <br>

The data can be found here: <br>
https://www.kaggle.com/code/hahmed747/intel-image-classification-fastai <br>

The model eventually reached a 95% training accuracy and a 75% testing accuracy. <br>
My first thought that this might be due to overfitting but after further data augmentation the model still failed to generalize. </b>
Upon observing the test set further I have found out that many images have the wrong label which lead to the wrong classifications. </br>

### Sidenotes:
I have impmented some helper and utility functions in scripts which can be found in the src folder. They serve the purpose of keeping the notebook as clean as possible and for validation. The notebook describes how to use the functions. 


## Next steps:
Work on generalization and overfitting.
More validation methods.
