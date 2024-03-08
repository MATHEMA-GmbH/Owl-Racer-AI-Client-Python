import pandas
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class OwlracerPreprocessor(Dataset):

    def __init__(self, data_path):
        # load the data
        self.data = pd.read_csv(data_path, sep=";")
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

    def clean_crashes(self, filter_size: int = 7):
        # clean crashes
        crashes = np.array(self.data["IsCrashed"] == True)
        kernel = np.ones(filter_size)
        crashes = np.pad(crashes, (int(filter_size / 2), 0))

        # window function
        result = np.convolve(crashes, kernel, mode='same')[int(filter_size / 2)::]
        result = result > 0

        self.data["IsCrashed"] = result
        self.data = self.data.loc[self.data["IsCrashed"] != True]
        self.data = self.data.loc[self.data["stepCommand"] != 0]

    def change_datatype(self, drop_columns: list = None):
        # change datatype
        if drop_columns is None:
            drop_columns = ["Id", "IsCrashed", "MaxVelocity", "Position.X", "Position.Y",
                            "PreviousCheckpoint", "Rotation", "Score", "ScoreOverall", "Ticks"]
        # self.data["Velocity"].replace(",", ".", inplace=True, regex=True)
        self.data["Velocity"] = self.data["Velocity"].replace(",", ".", regex=True)
        self.data["Velocity"] = pd.to_numeric(self.data["Velocity"])

        # drop unused tables
        self.data.drop(drop_columns, axis=1, inplace=True)
        self.data.drop(self.data.tail(1).index, inplace=True)

    def replace_stepcommand_labelmap(self, class2idx: dict):
        self.data["stepCommand"] = self.data["stepCommand"].replace(class2idx)

    def train_test_split(self, fixed_random_state: int = 444, test_size: float = 0.3):
        # generate data X and labels y
        X = self.data.drop("stepCommand", axis=1)
        y = self.data["stepCommand"]

        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=fixed_random_state)

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test


class OwlracerDataset(Dataset):

    def __init__(self, X, Y):
        self.X = torch.tensor(X.to_numpy(), dtype=torch.float32)

        self.y = torch.tensor(Y.to_numpy(), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.X[item], self.y[item]
