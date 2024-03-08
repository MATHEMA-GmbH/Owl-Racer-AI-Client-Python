import os
import sys
import argparse
import yaml

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import torch
from torch.utils.data import DataLoader
import mlflow

#import sclblonnx as so
import onnx

from owlracer_dataset import OwlracerDataset, OwlracerPreprocessor
from ModelPytorch.NN import NeuralNetwork
from ModelPytorch.ResNet import ResNet
from ModelPytorch.Exporter import ExporterToOnnx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="set the path of the data", required=True)
    parser.add_argument("--experiment", type=str, help="set mlflow experiment", required=True)
    args = parser.parse_args()
    return args


def set_mlflow_tracking(experiment_name):
    mlflow.set_experiment(experiment_name)
    active_run = mlflow.start_run(run_name=model_type)
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

    training_data = OwlracerDataset(X=X_train, Y=Y_train)
    testing_data = OwlracerDataset(X=X_test, Y=Y_test)

    return training_data, testing_data, preprocessor.Y_test


def mainLoop():
    if model_type == "NN":
        model = NeuralNetwork()

    elif model_type == "ResNet":
        model = ResNet(in_channels=params["in_channels"], out_channels=params["out_channels"])

    else:
        return

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-3,
                                                    total_steps=int((epochs + 1) * len(training_data) / batch_size))

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, model, loss, optimizer, scheduler)
        #if (t + 1) % 500 == 0:
        #    test_loop(test_loader, model, loss)

    print("Done!")

    test_loop(test_loader, model, loss)

    save_model(model)
    # 23.02.2024: Das hier auskommentiert:
    #transform_onnx()
    mlflow.end_run()


def train_loop(dataloader, model, loss_fn, optimizer, scheduler):
    # TODO: mlflow autologging for NN module not supported
    size = len(dataloader.dataset)
    running_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if batch % 100 == 0:
            current = batch * len(X)
            loss = (running_loss/(batch+1))
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            mlflow.log_metric("loss", loss, current)


def test_loop(dataloader, model, loss_fn):
    # TODO: mlflow autologging for NN module not supported
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    y_pred_list = []
    idx2class = {v: k for k, v in class2idx.items()}

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            y_pred_list.append(pred.argmax(1))

    y_pred_list = [idx2class[a.squeeze().tolist()] for a in y_pred_list]

    test_loss /= num_batches
    correct /= size
    accuracy = (100*correct)
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("test_loss", test_loss)

    y_test.replace(idx2class, inplace=True)

    report = classification_report(y_test, y_pred_list, output_dict=True)

    mlflow.log_dict(report, "classification_report")

    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list)).rename(columns=idx2class, index=idx2class)
    plot = sns.heatmap(confusion_matrix_df, annot=True)
    fig_name = os.path.join("train_pytorch_results", "confusion_matrix.png")
    figure = plot.get_figure()
    figure.savefig(fig_name, dpi=600)

    mlflow.log_artifact(fig_name, "confusion_matrix")


def save_model(model):
    mlflow.pytorch.log_model(model, artifact_path=model_type)

    exporter = ExporterToOnnx()
    onx = exporter.export_to_onnx(model, next(iter(test_loader)), os.path.join("train_pytorch_results", f"{model_type}-original.onnx"), target_opset)
    mlflow.onnx.log_model(onnx_model=onx, artifact_path=f"{model_type}-original.onnx")


def transform_onnx():
    graph = so.graph_from_file(os.path.join("train_pytorch_results", f"{model_type}-original.onnx"))
    # rename output to be like sklearn
    so.output.rename_output(graph, "output", "output_probability")

    # add node to add output_label
    n1 = so.node('ArgMax',
                 axis=1,
                 keepdims=1,
                 inputs=['output_probability'],
                 outputs=['output_label'])
    graph = so.add_node(graph, n1)
    graph = so.add_output(graph, 'output_label', "INT64", [1, 1])
    so.graph_to_file(graph, os.path.join("train_pytorch_results", f"{model_type}.onnx"))

    onnx_model = onnx.load(os.path.join("train_pytorch_results", f"{model_type}.onnx"))
    mlflow.onnx.log_model(onnx_model=onnx_model, artifact_path=f"{model_type}.onnx")


if __name__ == '__main__':
    args = parse_args()
    data_path = args.data

    class2idx = yaml.safe_load(open("examples/train/labelmap.yaml"))["class2idx"]
    params = yaml.safe_load(open("examples/train/params.yaml"))["train-pytorch"]
    batch_size = params["batch_size"]
    learning_rate = float(params["learning_rate"])
    epochs = params["epochs"]
    model_type = params["model_type"]
    target_opset = params["target_opset"]

    all_model_types = ["NN", "ResNet"]

    if model_type not in all_model_types:
        sys.stderr.write(f"Parameter error: model_type must be in {all_model_types}\n")
        sys.stderr.write("\tpython src/train_pytorch.py\n")
        sys.exit(1)

    set_mlflow_tracking(experiment_name=args.experiment)
    os.makedirs("train_pytorch_results", exist_ok=True)

    training_data, testing_data, y_test = get_features()
    train_loader = DataLoader(training_data, batch_size=batch_size)
    test_loader = DataLoader(testing_data)

    mainLoop()

