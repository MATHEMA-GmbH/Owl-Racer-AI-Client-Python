import os
import sys

import argparse
import yaml
import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from ray.tune.integration.mlflow import mlflow_mixin
import mlflow

from owlracer_dataset import OwlracerPreprocessor


def prase_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="set the path of the data", required=True)
    parser.add_argument("--experiment", type=str, help="set mlflow experiment", required=True)
    args = parser.parse_args()
    return args


def set_mlflow_tracking(experiment_name):
    mlflow.set_experiment(experiment_name)
    active_run = mlflow.start_run(run_name=classifier)
    run_id = active_run.info.run_id
    print(f"active mlflow run with id: {run_id}")

    mlflow.set_tag("data_set", data_path)

    mlflow.log_params(parameters)
    mlflow.log_param("target_opset", params["target_opset"])
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

@mlflow_mixin
def fit_model(model):
    mlflow.autolog(log_input_examples=True)
    model.fit(X_train, Y_train)

    return model


def train_decision_tree():
    model = DecisionTreeClassifier(**parameters)

    print("start fit")
    start = time.time()
    model= fit_model(model)
    end = time.time()
    print("Fit Time:", end - start)

    return model


def train_random_forest():
    model = RandomForestClassifier(**parameters)

    print("start fit")
    start = time.time()
    model = fit_model(model)
    end = time.time()
    print("Fit Time:", end - start)

    return model


def evaluate_model():
    mlflow.sklearn.eval_and_log_metrics(model=model, X=X_test, y_true=Y_test, prefix=f"test_")

    idx2class = {v: k for k, v in class2idx.items()}
    preds = model.predict(X_test)
    preds = [idx2class[pred] for pred in preds]

    Y_test.replace(idx2class, inplace=True)
    #report = classification_report(Y_test, preds, output_dict=True)
    print(classification_report(Y_test, preds))


def save_model(file_name):
    # https://onnx.ai/sklearn-onnx/
    initial_inputs = [('input', FloatTensorType([None, X_train.shape[1]]))]
    #onx = convert_sklearn(model, initial_types=initial_inputs, target_opset=params["target_opset"], verbose=1)
    onx = to_onnx(model,
                  initial_types=initial_inputs,
                  target_opset=params["target_opset"],
                  options={id(model): {'zipmap': False}},
                  final_types=[('output_label', Int64TensorType([None])),
                               ('output_probability', FloatTensorType([None, len(class2idx)]))])

    mlflow.onnx.log_model(onnx_model=onx, artifact_path=f"{file_name}.onnx")


if __name__ == '__main__':
    args = prase_args()
    data_path = args.data

    class2idx = yaml.safe_load(open("examples/train/labelmap.yaml"))["class2idx"]
    params = yaml.safe_load(open("examples/train/params.yaml"))["train-sklearn"]
    parameters = params["parameters"]

    classifier_list = ["DecisionTreeClassifier", "RandomForestClassifier"]
    classifier = params["classifier"]
    if classifier not in classifier_list:
        sys.stderr.write(f"Parameter error: classifier must be in {classifier_list}\n")
        sys.stderr.write("\tpython src/train_sklearn.py\n")
        sys.exit(1)

    set_mlflow_tracking(experiment_name=args.experiment)

    X_train, X_test, Y_train, Y_test = get_features()

    if classifier == "DecisionTreeClassifier":
        model = train_decision_tree()

    elif classifier == "RandomForestClassifier":
        model = train_random_forest()

    else:
        sys.exit(1)

    save_model(classifier)
