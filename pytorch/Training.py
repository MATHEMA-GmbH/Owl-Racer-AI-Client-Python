import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


from Owlracer_Dataset import Owlracer_Dataset
from Model import NN, ResNet
from Model.export_onnx import export_to_onnx

import os
os.path.join()

def train_loop(dataloader, model, loss_fn, optimizer, sheduler):
    size = len(dataloader.dataset)
    running_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sheduler.step()

        running_loss += loss.item()

        if batch % 100 == 0:
            current = batch * len(X)
            print(f"loss: {(running_loss/(batch+1)):>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def mainLoop():

    training_data = Owlracer_Dataset(train=True)
    batch_size = 128
    train_loader = DataLoader(training_data, batch_size=batch_size)

    test_data = Owlracer_Dataset(train=False)
    test_loader = DataLoader(test_data)

    learning_rate = 1e-6
    epochs = 100

    model = NN.NeuralNetwork()
    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    sheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, total_steps= int((epochs+1) * len(training_data)/batch_size))

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, model, loss, optimizer, sheduler)
        if (t + 1) % 500 == 0:
            test_loop(test_loader, model, loss)

    print("Done!")

    export_to_onnx(model, next(iter(test_loader)), "./trainedModels/DNN.onnx")


if __name__ == '__main__':
    mainLoop()

