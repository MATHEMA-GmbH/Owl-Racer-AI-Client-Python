import sys

import yaml
import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
import mlflow

from shared import (parse_args,
                    set_mlflow_tracking,
                    get_features,
                    generate_labelmap,
                    get_model_config)

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

    idx2class = {v: k for k, v in labelmap["class2idx"].items()}
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
                               ('output_probability', FloatTensorType([None, len(labelmap["class2idx"])]))])

    mlflow.onnx.log_model(onnx_model=onx, artifact_path=f"{file_name}.onnx")


if __name__ == '__main__':

    # --- read args, params, etc. ---
    args = parse_args()
    data_path = args.data

    params = yaml.safe_load(open("examples/train/params.yaml"))["train-sklearn"]
    parameters = params["parameters"]

    model_config = get_model_config("examples/train/model_config.yaml")

    classifier_list = ["DecisionTreeClassifier", "RandomForestClassifier"]
    model_type = params["model_type"]
    if model_type not in classifier_list:
        sys.stderr.write(f"Parameter error: model_type must be in {classifier_list}\n")
        sys.stderr.write("\tpython src/train_sklearn.py\n")
        sys.exit(1)
    # --- \ ----

    # --- ---
    labelmap = generate_labelmap(labelmap_path="examples/train/labelmap.yaml",
                                 change_commands=model_config["change_commands"])

    X_train, X_test, Y_train, Y_test = (
        get_features(data_path=data_path,
                     class2idx=labelmap["class2idx"],
                     model_config=model_config))
    # --- \ ---

    # --- Here the parameters for the tracking are set and the tracking is started ---
    set_mlflow_tracking(experiment_name=args.experiment,
                        model_type=model_type,
                        data_path=data_path,
                        params=params,
                        labelmap=labelmap,
                        model_config=model_config,
                        server_log=args.server_log,
                        logging_debug=args.logging_debug)
    # --- \ ---

    # --- The actual training ---
    if model_type == "DecisionTreeClassifier":
        model = train_decision_tree()
    elif model_type == "RandomForestClassifier":
        model = train_random_forest()
    else:
        sys.exit(1)
    # --- \ ---

    # --- Model gets converted to onnx and logged with mlflow
    save_model(model_type)
    # --- \ ---
