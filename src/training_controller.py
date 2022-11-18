import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
from .helper_funcs import get_lr, calculate_accuracy
from datetime import datetime
import timeit
import numpy as np

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, model, train_location, test_location,
                 transform, eval_transform, batch_size, loss_function, optimizer, epochs, eval_location=None,
                 scheduler=None, gradient_clipping=False):

        self._model = model
        self._epochs = epochs
        self._loss_function = loss_function
        self._scheduler = scheduler
        self._optimizer = optimizer
        self._transform = transform
        self._train_location = train_location
        self._test_location = test_location
        self._eval_location = eval_location
        self._gradient_clipping = gradient_clipping
        self._batch_size = batch_size
        self._losses = []
        self._accs = []

        self._trainset = torchvision.datasets.ImageFolder(root=train_location, transform=transform)
        self._trainloader = torch.utils.data.DataLoader(self._trainset, batch_size=self._batch_size,
                                                        shuffle=True, pin_memory=True)

        self._testset = torchvision.datasets.ImageFolder(root=test_location, transform=eval_transform)
        self._testloader = torch.utils.data.DataLoader(self._testset, batch_size=self._batch_size,
                                                       shuffle=True, pin_memory=True)

        if eval_location:
            self._evalset = torchvision.datasets.DatasetFolder(root=eval_location, transform=eval_transform)
            self._evaloader = torch.utils.data.DataLoader(self._evalset, batch_size=self._batch_size,
                                                          shuffle=True, pin_memory=True)

        self._model.to(device)

    def train(self):
        print("Training on ", torch.cuda.get_device_name(0))
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"Started training at: {current_time}")

        for epoch in range(self._epochs):
            running_loss = 0.0
            self._model = self._model.train()

            for i, batch in enumerate(self._trainloader):
                self._optimizer.zero_grad()

                images, targets = batch
                images, targets = images.to(device), targets.to(device)

                out = self._model(images)
                loss = self._loss_function(out, targets)
                loss.backward()

                self._optimizer.step()

                if self._gradient_clipping:
                    nn.utils.clip_grad_value_(self._model.parameters(), 0.1)

                running_loss += loss.item()

            if self._scheduler:
                self._scheduler.step()

            self._model = self._model.eval()
            acc = calculate_accuracy(self._trainloader, self._model)
            loss = running_loss / len(self._trainloader)
            self._losses.append(loss)
            self._accs.append(acc * 0.1)

            now = datetime.now()

            print(f"Epoch [{epoch}]: loss: {loss}, time finished: {now}, learning "
                  f"rate: {get_lr(self._optimizer)}, train acc {acc} %")

    def test_dataloader_speed(self):

        """
        Function to determine most efficient numbers of workers
        num workers > 0 slowed down the training process drastically. Probably due to windows OS.
        """
        arr = [12, 20, 22]

        for i in arr:

            self._trainloader = torch.utils.data.DataLoader(self._trainset, batch_size=32,
                                                            shuffle=True, num_workers=i)
            starttime = timeit.default_timer()
            for epoch in range(8):
                for i, batch in enumerate(self._trainloader):
                    pass

            print(f"Num workers: {i}, time taken {starttime - timeit.default_timer()}")

    def visualize_incorrectly_classified_images(self, n):

        """
        :param dataloader: dataloader from which the data is drawn from
        :param model: model to test on
        :param n: number of random images to be visualized
        """
        wrong_images = []
        wrong_labels = []

        with torch.no_grad():
            for images, labels in self._testloader:
                images, labels = images.to('cuda'), labels.to('cuda')
                outputs = self._model(images)
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

    def visualize_results(self):

        plt.plot(self._accs, label="Accuracy")
        plt.plot(self._losses, label="Loss")
        plt.legend()
        plt.show()

    def getModel(self):
        return self._model

    def setModel(self, model):
        self._model = model
