import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
from . helper_funcs import get_lr, calculate_accuracy
from datetime import datetime
import timeit

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, model, train_location, test_location, eval_location,
                 transform, eval_transform, batch_size, loss_function, optimizer, epochs, scheduler=None, gradient_clipping=False):

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

        self._evalset = torchvision.datasets.ImageFolder(root=eval_location, transform=eval_transform)
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
            self._accs.append(acc*0.1)

            now = datetime.now()

            print(f"Epoch [{epoch}]: loss: {loss}, time finished: {now}, learning "
                  f"rate: {get_lr(self._optimizer)}, train acc {acc} %")
            running_loss = 0.0

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

    def visualize_results(self):

        plt.plot(self._accs, label="Accuracy")
        plt.plot(self._losses, label="Loss")
        plt.legend()
        plt.show()

    def getModel(self):
        return self._model