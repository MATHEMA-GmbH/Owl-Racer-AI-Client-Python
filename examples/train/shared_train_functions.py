import argparse
import os
import mlflow
import logging
import yaml

from owlracer_dataset import OwlracerPreprocessor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="set the path of the data", required=True)
    parser.add_argument("--experiment", type=str, help="set mlflow experiment", required=True)
    parser.add_argument("--server-log",
                        action="store_true",
                        help="set logging to server; server is specified in params.yaml.",
                        required=False)
    parser.add_argument("--logging-debug",
                        action="store_true",
                        help="enable logging debug for mlflow-server.",
                        required=False)
    parser.add_argument("--replace_commands",
                        action="store_true",
                        help="enables command replacement as specified in \"datachange.yaml\".",
                        required=False)
    args = parser.parse_args()
    return args

def set_mlflow_tracking(experiment_name,
                        model_type,
                        data_path,
                        params,
                        labelmap,
                        datachange,
                        replace_commands,
                        server_log,
                        logging_debug):

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

    active_run = mlflow.start_run(run_name=model_type)
    run_id = active_run.info.run_id
    print(f"active mlflow run with id: {run_id}")

    mlflow.set_tag("data_set", data_path)

    mlflow.set_tag("repository", os.getenv("MLFLOW_PROJECT_ENV"))
    mlflow.log_params(params)
    mlflow.log_param("target_opset", params["target_opset"])
    mlflow.log_dict(labelmap, "labelmap.yaml")
    mlflow.log_dict(datachange, "datachange.yaml")
    mlflow.log_param("replace_commands", replace_commands)

    print(f"mlflow artifact uri: {mlflow.get_artifact_uri()}")
    print(f"mlflow tracking uri: {mlflow.get_tracking_uri()}")

    if logging_debug:
        logging.getLogger("mlflow").setLevel(logging.DEBUG)

def get_features(data_path, class2idx, datachange, replace_commands):
    preprocessor = OwlracerPreprocessor(data_path=data_path)
    preprocessor.clean_crashes()
    preprocessor.clean_prestart_data()
    preprocessor.change_datatype(used_columns=datachange["used_columns"])
    preprocessor.change_commands(datachange=datachange, replace_commands=replace_commands)
    preprocessor.replace_stepcommand_labelmap(class2idx=class2idx)
    preprocessor.train_test_split()

    X_train = preprocessor.X_train
    X_test = preprocessor.X_test
    Y_train = preprocessor.Y_train
    Y_test = preprocessor.Y_test

    return X_train, X_test, Y_train, Y_test

def get_labelmap(labelmap_path):
    with open(labelmap_path, "r") as yaml_file:
        labelmap = yaml.safe_load(yaml_file)
        labelmap.update({"idx2class": {y: x for x,y in labelmap["class2idx"].items()}})
    with open(labelmap_path, "w") as yaml_file:
        yaml_file.write("# idx2class is automatically derived from class2idx. Only edit class2idx.\n")
        yaml.safe_dump(labelmap, yaml_file)
    return labelmap

def get_datachange(datachange_path):
    with open(datachange_path, "r") as yaml_file:
        datachange = yaml.safe_load(yaml_file)
    return datachange