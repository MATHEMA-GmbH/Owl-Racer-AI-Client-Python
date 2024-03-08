import argparse
import os
import yaml

import time
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV

from skl2onnx.common.data_types import StringTensorType, Int64TensorType, FloatTensorType
from skl2onnx import convert_sklearn, to_onnx

# from tune_sklearn import TuneGridSearchCV

# from ray.tune.integration.mlflow import mlflow_mixin
import mlflow

from owlracer_dataset import OwlracerPreprocessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="set the path of the data", required=True)
    parser.add_argument("--experiment", type=str, help="set mlflow experiment", required=True)
    args = parser.parse_args()
    return args


def get_models_pipeline():
    models = [{"experiment_name": "RF",
               "classifier": make_pipeline(StandardScaler(), RandomForestClassifier()),
               "parameter_grid": {"randomforestclassifier__n_estimators": [n for n in range(10, 200, 10)],
                                  "randomforestclassifier__criterion": ("entropy", "gini"),
                                  "randomforestclassifier__min_samples_split": [n for n in range(2, 21, 2)],
                                  "randomforestclassifier__max_depth": [n for n in range(50, 201, 50)]}},
              {"experiment_name": "DT",
               "classifier": make_pipeline(StandardScaler(), DecisionTreeClassifier()),
               "parameter_grid": {"decisiontreeclassifier__criterion": ("gini", "entropy"),
                                  "decisiontreeclassifier__max_depth": [n for n in range(50, 301, 50)]}},
              {"experiment_name": "KNN",
               "classifier": make_pipeline(StandardScaler(), PCA(), KNeighborsClassifier()),
               "parameter_grid": {"pca__n_components": [0.95, 0.85, 0.75, 0.6, 0.5],
                                  "kneighborsclassifier__n_neighbors": [n for n in range(1, 41, 5)],
                                  "kneighborsclassifier__weights": ['uniform', 'distance']}},
              {"experiment_name": "SVC",
               "classifier": make_pipeline(StandardScaler(), SVC()),
               "parameter_grid": {"svc__C": [0.5, 1.0], "svc__kernel": ["linear", "poly", "rbf", "sigmoid"]}}
              ]

    return models


def set_mlflow_tracking(run_name):
    active_run = mlflow.start_run(run_name=run_name)
    run_id = active_run.info.run_id
    print(f"active mlflow run with id: {run_id}")

    mlflow.set_tag("data_set", data_path)

    mlflow.log_params(params)
    mlflow.log_dict(class2idx, "labelmap-class2idx")


def get_features():
    preprocessor = OwlracerPreprocessor(data_path=data_path)
    preprocessor.clean_crashes()
    preprocessor.change_datatype()
    preprocessor.replace_stepcommand_labelmap(class2idx=class2idx)
    preprocessor.train_test_split()

    X_train = preprocessor.X_train
    X_test = preprocessor.X_test
    Y_train = preprocessor.Y_train
    Y_test = preprocessor.Y_test

    return X_train, X_test, Y_train, Y_test

# @mlflow_mixin
def fit_model(model, X_train, Y_train, experiment):
    set_mlflow_tracking(experiment)
    mlflow.autolog(log_input_examples=True)
    model.fit(X_train, Y_train)

    return model


def train_models(mlflow_experiment):
    # Train models
    local_dir = os.path.join(os.getcwd(), "train_hyperopt_results")
    os.makedirs(local_dir, exist_ok=True)

    mlflow.set_experiment(mlflow_experiment)

    # if windows, datetime needs to be removed, cannot work with numbers in path -.-
    # and comment out name=selected_model + local_dir=local_dir

    for model in models:
        selected_model = str(type(model.get("classifier")[-1])).split('.')[3][0: -2] \
             + "-" + datetime.now().strftime("%d%m%y%H%M%S%f")

        print(selected_model)
        # https://docs.ray.io/en/master/tune/tutorials/tune-sklearn.html
        # trained_model = TuneGridSearchCV(
        #     estimator=model.get("classifier"),
        #     param_grid=model.get("parameter_grid"),
        #     scoring="jaccard_weighted",
        #     name=selected_model,
        #     local_dir=local_dir
        # )

        trained_model = GridSearchCV(
            estimator=model.get("classifier"),
            param_grid=model.get("parameter_grid"),
            scoring="jaccard_weighted"
        )

        print("start training")
        trained_model = fit_model(trained_model, X_train, Y_train, model.get("experiment_name"))

        best_params = trained_model.best_params_
        print(best_params)
        for key in best_params:
            print(f"{key}: {best_params[key]}")

        save_model_path = os.path.join("dvc_data", "hyperopt", selected_model)
        os.makedirs(save_model_path, exist_ok=True)

        evaluate_model(trained_model.best_estimator)

        save_model(selected_model, trained_model.best_estimator)

        mlflow.end_run()


def evaluate_model(model):
    mlflow.sklearn.eval_and_log_metrics(model=model, X=X_test, y_true=Y_test, prefix=f"test_")

    idx2class = {v: k for k, v in class2idx.items()}
    preds = model.predict(X_test)
    preds = [idx2class[pred] for pred in preds]
    Y_test.replace(idx2class, inplace=True)

    print(classification_report(Y_test, preds))



def save_model(file_name, model):
    # https://onnx.ai/sklearn-onnx/
    initial_inputs = [('input', FloatTensorType([None, X_train.shape[1]]))]
    #onx = convert_sklearn(model, initial_types=initial_inputs)
    onx = to_onnx(model,
                  initial_types=initial_inputs,
                  target_opset=params["target_opset"],
                  options={id(model): {'zipmap': False}},
                  final_types=[('output_label', Int64TensorType([None])),
                               ('output_probability', FloatTensorType([None, len(class2idx)]))])

    mlflow.onnx.log_model(onnx_model=onx, artifact_path=f"{file_name}.onnx")


if __name__ == '__main__':
    args = parse_args()
    data_path = args.data

    params = yaml.safe_load(open("examples/train/params.yaml"))["hyperopt"]
    class2idx = yaml.safe_load(open("examples/train/labelmap.yaml"))["class2idx"]

    X_train, X_test, Y_train, Y_test = get_features()

    models = get_models_pipeline()
    train_models(args.experiment)
