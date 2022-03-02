import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class Owlracer_Dataset(Dataset):

    def __init__(self, train=True):
        # load the data
        data = pd.read_csv("../data/Data_Track2_1108.csv", sep=";")

        # clean crashes
        crashes = np.array(data["IsCrashed"] == True)
        filter_size = 7
        kernel = np.ones(filter_size)
        crashes = np.pad(crashes, (int(filter_size / 2), 0))

        # window function
        result = np.convolve(crashes, kernel, mode='same')[int(filter_size / 2)::]
        result = result > 0

        data["IsCrashed"] = result
        data = data.loc[data["IsCrashed"] != True]
        data = data.loc[data["stepCommand"] != 0]

        # change datatype
        data["Velocity"].replace(",", ".", inplace=True, regex=True)
        data["Velocity"] = pd.to_numeric(data["Velocity"])

        # drop unused tables
        data.drop(
            ["Id", "IsCrashed", "MaxVelocity", "Position.X", "Position.Y", "PreviousCheckpoint", "Rotation",
             "Score", "ScoreOverall", "Ticks"], axis=1, inplace=True)
        data.drop(data.tail(1).index, inplace=True)

        # generate data X and labels y
        X = data.drop("stepCommand", axis=1)
        y = data["stepCommand"]
        fixed_random_state = 444

        # generate train and test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=fixed_random_state)

        if train:
            self.X = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
            self.y = torch.tensor(Y_train.to_numpy(), dtype=torch.long)
        else:
            self.X = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
            self.y = torch.tensor(Y_test.to_numpy(), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

