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
    args = parser.parse_args()
    return args

def set_mlflow_tracking(experiment_name,
                        model_type,
                        data_path,
                        params,
                        labelmap,
                        model_config,
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
    mlflow.log_dict(model_config, "model_config.yaml")

    print(f"mlflow artifact uri: {mlflow.get_artifact_uri()}")
    print(f"mlflow tracking uri: {mlflow.get_tracking_uri()}")

    if logging_debug:
        logging.getLogger("mlflow").setLevel(logging.DEBUG)

def get_features(data_path, class2idx, model_config):
    preprocessor = OwlracerPreprocessor(data_path=data_path)
    preprocessor.clean_crashes()
    preprocessor.clean_prestart_data() # remove data from before the start of the game
    preprocessor.change_datatype(
        used_features=model_config["features"]["used_features"],
        normalization_constants = model_config["features"]["normalization_constants"])
    if (model_config["change_commands"]["replace_commands"]["commands"] != None) and (model_config["change_commands"]["replace_commands"]["before_training"] == True):
        preprocessor.replace_commands(command_replacement=model_config["change_commands"]["replace_commands"]["commands"])
    if model_config["change_commands"]["drop_commands"] != None:
        preprocessor.drop_unused_commands(drop_commands=model_config["change_commands"]["drop_commands"])
    preprocessor.replace_stepcommand_labelmap(class2idx=class2idx)
    preprocessor.train_test_split()

    X_train = preprocessor.X_train
    X_test = preprocessor.X_test
    Y_train = preprocessor.Y_train
    Y_test = preprocessor.Y_test

    return X_train, X_test, Y_train, Y_test

def generate_labelmap(labelmap_path, change_commands):
    # generates labelmap with class2idx and idx2class according to the used_commands
    with open(labelmap_path, "w+") as yaml_file:
        class2idx = {"class2idx": {}}
        drop_commands = [] # contains commands that are dropped and thus not used for training
        if change_commands["drop_commands"] != None:
            # add commands to drop-list that shall not be used for training
            drop_commands = drop_commands + change_commands["drop_commands"]
        if (change_commands["replace_commands"]["commands"] != None) and (change_commands["replace_commands"]["before_training"] == True):
            # add commands to drop-list that shall be replaced and thus not be used for training
            drop_commands = drop_commands + list(change_commands["replace_commands"]["commands"].keys())
        used_commands = [x for x in range(7) if x not in drop_commands] # right now: 7 possible commands [0,1,..,6]
        # generate labelmap:
        for i in range(len(used_commands)):
            class2idx["class2idx"][used_commands[i]] = i
        idx2class = {"idx2class": {y: x for x,y in class2idx["class2idx"].items()}}
        yaml_file.write("# This file is automatically generated. Do not edit!\n")
        yaml.safe_dump(class2idx, stream=yaml_file)
        yaml.safe_dump(idx2class, stream=yaml_file)
    labelmap = yaml.safe_load(open(labelmap_path, "r"))
    return labelmap

def get_model_config(config_path):
    with open(config_path, "r") as yaml_file:
        model_config = yaml.safe_load(yaml_file)
    return model_config