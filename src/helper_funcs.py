import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def calculate_accuracy(dataloader, model):

    """
    :param dataloader: dataloader you want to measure the accuracy on
    :param model: Your model
    :return: accuracy as an int
    """
    model.eval()
    correct_images = 0
    total_images = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            correct_images += (predicted == labels).sum().item()
        acc = 100 * correct_images // total_images
        return acc


def res_block(channels, resize):
    if resize:
        layers = [nn.Conv2d(channels[0],
                            channels[1],
                            kernel_size=(3, 3),
                            stride=(2, 2),
                            padding=1),
                  nn.BatchNorm2d(channels[1]),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(channels[1],
                            channels[2],
                            kernel_size=(1, 1),
                            stride=(1, 1),
                            padding=0),
                  nn.BatchNorm2d(channels[2])]

    else:
        layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=channels[0],
                            out_channels=channels[1],
                            kernel_size=(1, 1),
                            stride=(1, 1),
                            padding=0),
            torch.nn.BatchNorm2d(channels[1]),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=channels[1],
                            out_channels=channels[0],
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
            torch.nn.BatchNorm2d(channels[0]))

    return nn.Sequential(*layers)


def conv_block(input_size, output_size, pool, filter_size, padding=0, stride=(1, 1)):
    layers = [nn.Conv2d(input_size, output_size, kernel_size=filter_size, padding=padding, stride=stride),
              nn.BatchNorm2d(output_size)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class ResidualBlock(nn.Module):

    def __init__(self, channels, resize):

        """
            Module class for a single residual block.
            "resize" denotes a boolean which determines whether the input should be scaled to a different output size.

            "channels" is an array which contains up to 3 different integers which determine the input and output dimensions.
            Example: channels[0] = input dimension, channels[1] = intermediate scaling, channels[2] = output dimension

            In case of resize == False:
            channels[0] = input dimension, channels[1] = output dimension, channels[2] = unused
        """

        super().__init__()

        self.resize = resize
        self._block = res_block(channels, self.resize)

        if resize:
            self._shortcut = conv_block(channels[0], channels[2], False, (1, 1), padding=0, stride=(2, 2))

    def forward(self, x):

        if not self.resize:
            sc = x
        else:
            sc = self._shortcut(x)

        block = self._block(x)
        x = torch.nn.functional.relu(block + sc)

        return x
