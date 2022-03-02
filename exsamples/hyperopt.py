import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from skl2onnx.common.data_types import Int64TensorType
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from tune_sklearn import TuneGridSearchCV, TuneSearchCV


# Important for ray on windows, not usabel in venv
# see https://github.com/ray-project/ray/issues/13794

def convert_dataframe_schema_for_onnx(df, drop=None):
    inputs = []
    for k, v in zip(df.columns, df.dtypes):
        if drop is not None and k in drop:
            continue
        if v == 'int64':
            t = Int64TensorType([None, 1])
        elif v == 'float64':
            t = FloatTensorType([None, 1])
        else:
            t = StringTensorType([None, 1])
        k = k.replace(".","_")
        inputs.append((k, t))
    return inputs

def change_to_float(value):
    return value.astype(np.float32)

def main():
    # load the data
    data = pd.read_csv("./data/Data_Track2_1108.csv", sep=";")


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
    data["Velocity"].replace(",",".", inplace=True, regex=True)
    data["Velocity"] = pd.to_numeric(data["Velocity"])


    # drop unsused tables
    data.drop(
        ["Id", "IsCrashed", "MaxVelocity", "Position.X", "Position.Y", "PreviousCheckpoint", "Rotation",
         "Score", "ScoreOverall", "Ticks"], axis=1, inplace=True)
    data.drop(data.tail(1).index, inplace=True)

    # generate data X and labels y
    X = data.drop("stepCommand", axis=1)
    y = data["stepCommand"]

    # generate train and test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=444)

    print("Train Data")
    print(classification_report(Y_train, Y_train))
    print("Test Data")
    print(classification_report(Y_test, Y_test))

    models =[{"classifier": make_pipeline(StandardScaler(), RandomForestClassifier()),
            "parameter_grid": {"randomforestclassifier__n_estimators": [ n for n in range(10, 200, 10)], "randomforestclassifier__criterion": ("entropy", "gini"),
            "randomforestclassifier__min_samples_split": [ n for n in range(2, 21, 2)], "randomforestclassifier__max_depth": [ n for n in range(50, 201, 50 )]}},
            {"classifier": make_pipeline(StandardScaler(), DecisionTreeClassifier()),
            "parameter_grid": {"decisiontreeclassifier__criterion": ("gini", "entropy"), "decisiontreeclassifier__max_depth": [ n for n in range(50,301, 50)]}},
            {"classifier": make_pipeline(StandardScaler(), PCA(), KNeighborsClassifier()),
            "parameter_grid": {"pca__n_components": [0.95, 0.85, 0.75, 0.6, 0.5], "kneighborsclassifier__n_neighbors": [n for n in range(1,41, 5)], "kneighborsclassifier__weights": ['uniform', 'distance']}},
            {"classifier": make_pipeline(StandardScaler(), SVC()),
            "parameter_grid": {"svc__C": [0.5, 1.0], "svc__kernel": ["linear", "poly", "rbf", "sigmoid"]}}
            ]

    trained_models = []

    # Train models
    for model in models:
        print(str(type(model.get("classifier")[-1])).split('.')[3][0: -2])
        # https://docs.ray.io/en/master/tune/tutorials/tune-sklearn.html
        trained_model = TuneGridSearchCV(
            estimator=model.get("classifier"),
            param_grid=model.get("parameter_grid"),
            scoring="jaccard_weighted"
            )

        trained_model.fit(X_train, Y_train)


        print(trained_model.best_params_)

        # evaluate
        preds = trained_model.predict(X_test)
        print(classification_report(Y_test, preds))

        # https://onnx.ai/sklearn-onnx/
        initial_inputs = convert_dataframe_schema_for_onnx(X_train)
        onx = convert_sklearn(trained_model.best_estimator, initial_types=initial_inputs)
        with open("./trainedModels/{0}_owlracer.onnx".format(str(type(model.get("classifier")[1])).split('.')[3][0: -2]), "wb") as f:
            f.write(onx.SerializeToString())

        trained_models.append(model)

if __name__ == '__main__':
    main()
