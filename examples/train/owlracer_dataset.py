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

    def clean_prestart_data(self):
        # delete rows where game has not started yet
        self.data = self.data[self.data["ScoreStep"] != 0]

    def replace_commands(self, command_replacement):
        self.data["stepCommand"] = self.data["stepCommand"].replace(command_replacement)

    def drop_unused_commands(self, drop_commands):
        # only keep rows if the respective command should be used for training
        for k in drop_commands:
            self.data = self.data[self.data["stepCommand"] != k]


    def change_datatype(self, used_features, normalization_constants):
        # formatting velocity feature
        self.data["Velocity"] = self.data["Velocity"].replace(",", ".", regex=True)
        self.data["Velocity"] = pd.to_numeric(self.data["Velocity"])

        features = list(self.data.columns.values)
        # get features that shall not be used for training
        drop_features = [x for x in features if (x not in used_features) and (x != "stepCommand")]

        # drop unused tables
        self.data = self.data.drop(drop_features, axis=1)
        self.data = self.data.drop(self.data.tail(1).index)

        # normalization
        for feature_name, normalization_constant in normalization_constants.items():
            if feature_name in self.data.columns:
                self.data[feature_name] = self.data[feature_name].apply(lambda x: x*normalization_constant)

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
