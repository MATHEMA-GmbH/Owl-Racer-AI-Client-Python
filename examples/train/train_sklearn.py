import logging
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
# from ray.tune.integration.mlflow import mlflow_mixin
import mlflow

from owlracer_dataset import OwlracerPreprocessor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="set the path of the data", required=True)
    parser.add_argument("--experiment", type=str, help="set mlflow experiment", required=True)
    parser.add_argument("--server-log",
                        action="store_true",
                        help="optional: set logging to server; server is specified in params.yaml",
                        required=False)
    parser.add_argument("--logging-debug",
                        action="store_true",
                        help="optional: enable logging debug for mlflow-server",
                        required=False)
    args = parser.parse_args()
    return args


def set_mlflow_tracking(experiment_name, server_log, logging_debug, drop_columns: list, used_columns: list, used_commands: list, replace_commands: dict):

    if server_log:
        # logging to server specified in .env
        from config.config import settings

        os.environ['MLFLOW_TRACKING_URI'] = settings.REMOTE_SERVER_URI
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = settings.MLFLOW_S3_ENDPOINT_URL
        os.environ['AWS_ACCESS_KEY_ID'] = settings.AWS_ACCESS_KEY_ID
        os.environ['AWS_SECRET_ACCESS_KEY'] = settings.AWS_SECRET_ACCESS_KEY
        # os.environ['MLFLOW_PROJECT_ENV'] = settings.MLFLOW_PROJECT_ENV # maybe needed for logging the repo under tags

        # mlflow.set_tracking_uri(uri=remote_server_uri)
        # print(f"Logging to: {remote_server_uri}")
        print(f"Logging to: {settings.REMOTE_SERVER_URI}")

        if settings.MLFLOW_S3_BUCKET != None:
            # MLFLOW_S3_BUCKET is specified in .env
            if mlflow.get_experiment_by_name(experiment_name) == None:
                # experiment does not exist yet and thus can be created with specified artifact_location
                mlflow.create_experiment(name=experiment_name, artifact_location=settings.MLFLOW_S3_BUCKET)

    else:
        print("Logging locally")

    mlflow.set_experiment(experiment_name) # experiment will be created if not existing

    active_run = mlflow.start_run(run_name=classifier)
    run_id = active_run.info.run_id
    print(f"active mlflow run with id: {run_id}")

    mlflow.set_tag("data_set", data_path)

    mlflow.set_tag("repository", os.getenv("MLFLOW_PROJECT_ENV"))
    mlflow.log_params(parameters)
    mlflow.log_param("target_opset", params["target_opset"])
    mlflow.log_param("drop columns", drop_columns)
    mlflow.log_param("used columns", used_columns)
    mlflow.log_param("used commands", used_commands)
    mlflow.log_param("replace commands", replace_commands)
    mlflow.log_dict(class2idx, "labelmap-class2idx.yaml")

    print(f"mlflow artifact uri: {mlflow.get_artifact_uri()}")
    print(f"mlflow tracking uri: {mlflow.get_tracking_uri()}")

    if logging_debug:
        logging.getLogger("mlflow").setLevel(logging.DEBUG)


def get_features(drop_columns: list = None, replace_commands = None):
    preprocessor = OwlracerPreprocessor(data_path=data_path)
    preprocessor.clean_crashes()
    preprocessor.clean_prestart_data()
    [used_columns, used_commands, replace_commands] = preprocessor.change_datatype(drop_columns=drop_columns, replace_commands=replace_commands)
    preprocessor.replace_stepcommand_labelmap(class2idx=class2idx)
    preprocessor.train_test_split()

    X_train = preprocessor.X_train
    X_test = preprocessor.X_test
    Y_train = preprocessor.Y_train
    Y_test = preprocessor.Y_test

    return used_columns, used_commands, replace_commands, X_train, X_test, Y_train, Y_test

# @mlflow_mixin
def fit_model(model):
    mlflow.sklearn.autolog(log_input_examples=True)
    model.fit(X_train, Y_train)
    # run_id = mlflow.last_active_run().info.run_id
    # print(f"run_id: {run_id}")

    return model


def train_decision_tree():
    # --- Class comes from sklearn.tree ---
    model = DecisionTreeClassifier(**parameters)

    print("start fit")
    start = time.time()
    model= fit_model(model)
    end = time.time()
    print("Fit Time:", end - start)

    return model


def train_random_forest():
    # --- Class comes from sklearn.ensemble ---
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
    print("Converting model to .onnx...")
    onx = to_onnx(model,
                  initial_types=initial_inputs,
                  target_opset=params["target_opset"],
                  options={id(model): {'zipmap': False}},
                  final_types=[('output_label', Int64TensorType([None])),
                               ('output_probability', FloatTensorType([None, len(class2idx)]))])

    mlflow.onnx.log_model(onnx_model=onx, artifact_path=f"{file_name}.onnx")


if __name__ == '__main__':

    # --- read args, params, etc. ---
    args = parse_args()
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
    # --- \ ----

    drop_columns = ["Time", "Id", "IsCrashed", "MaxVelocity", "Position.X", "Position.Y",
                    "Checkpoint", "Rotation", "ScoreStep", "ScoreOverall", "Ticks",
                    "Velocity"]
    replace_commands = {0:1, 2:1}

    # --- ---
    used_columns, used_commands, replace_commands, X_train, X_test, Y_train, Y_test = get_features(drop_columns=drop_columns, replace_commands=replace_commands)
    # --- \ ---

    # --- Here the parameters for the tracking are set and the tracking is started ---
    set_mlflow_tracking(experiment_name=args.experiment, server_log=args.server_log, logging_debug=args.logging_debug, drop_columns=drop_columns, used_columns=used_columns, used_commands=used_commands, replace_commands=replace_commands)
    # --- \ ---

    # --- The actual training ---
    if classifier == "DecisionTreeClassifier":
        model = train_decision_tree()
    elif classifier == "RandomForestClassifier":
        model = train_random_forest()
    else:
        sys.exit(1)
    # --- \ ---

    # --- Model gets converted to onnx and logged with mlflow
    save_model(classifier)
    # --- \ ---
