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


def visualize_incorrectly_classified_images(dataloader, model, n):

    """
    :param dataloader: dataloader from which the data is drawn from
    :param model: model to test on
    :param n: number of random images to be visualized
    """
    wrong_images = []
    wrong_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            for i in range(len(images)):
                if labels[i].item() is not predicted[i].item():
                    wrong_images.append(images[i])
                    wrong_labels.append({"predicted_label": predicted[i].item(),
                                         "actual_label": labels[i].item()})

    wrong_images = torch.stack(wrong_images)
    wrong_images = wrong_images.cpu()

    random_indexes = np.random.randint(len(wrong_images), size=n)
    columns = 5 if n >= 5 else n
    rows = n // 5 if n > 5 else 1
    fig, ax = plt.subplots(nrows=rows, ncols=columns, figsize=[50, 50])

    pred_translator = {
        0: "buildings",
        1: "forest",
        2: "glacier",
        3: "mountain",
        4: "sea",
        5: "street",
    }

    for i, axi in enumerate(ax.flat):
        img = wrong_images[random_indexes[i]]
        img = img.permute(1, 2, 0)

        axi.imshow(img)
        axi.set_title(f"Pred: {pred_translator[wrong_labels[i]['predicted_label']]},"
                      f" Label: {pred_translator[wrong_labels[i]['actual_label']]}", fontsize=40)

    plt.show()


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