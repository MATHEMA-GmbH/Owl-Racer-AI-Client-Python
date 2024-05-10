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

    def change_commands(self, datachange, replace_commands):
        if replace_commands == True:
            self.data["stepCommand"] = self.data["stepCommand"].replace(datachange["replace_commands"])
        else:
            # only keep rows if the respective command should be used for training
            commands = self.data["stepCommand"].drop_duplicates().to_list()
            drop_commands = [x for x in commands if x not in datachange["used_commands"]]
            for k in drop_commands:
                self.data = self.data[self.data["stepCommand"] != k]


    def change_datatype(self, used_columns):
        # change datatype
        self.data["Velocity"] = self.data["Velocity"].replace(",", ".", regex=True)
        self.data["Velocity"] = pd.to_numeric(self.data["Velocity"])

        columns = list(self.data.columns.values)
        drop_columns = [x for x in columns if (x not in used_columns) and (x != "stepCommand")]

        # drop unused tables
        self.data = self.data.drop(drop_columns, axis=1)
        self.data = self.data.drop(self.data.tail(1).index)

        # normalization
        for distance_name in ["Distance.Front", "Distance.FrontLeft", "Distance.FrontRight", "Distance.Left", "Distance.Right"]:
            if distance_name in self.data.columns:
                self.data[distance_name] = self.data[distance_name].apply(lambda x: x/1000)
        if "ScoreChange" in self.data.columns:
            self.data["ScoreChange"] = self.data["ScoreChange"].apply(lambda x: x/10)
        if "WrongDirection" in self.data.columns:
            self.data["WrongDirection"] = self.data["WrongDirection"].astype(int)

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
