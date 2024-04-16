![Logo](https://github.com/MATHEMA-GmbH/Owl-Racer-AI/blob/main/doc/owlracer-logo.png?raw=true)

# The Python API
___
## Content
This folder contains all files neccacery to build and compile a working Python Api
for the Owlracer server. Further the Folder includes easy to understand and some advanced examples.
Displayed below the content is depicted.

```
.   
├── README.md (This file)    
├── data (Data captured for supervised learning)  
├── examples (Easy to follow and extend with scripts for training and playing) 
├── jupyter_deprecated (example Python Notebooks, will be updated for new version)  
├── lib (Package for building the python API)         
├── trainedModels (trained models with different algorithms and frameworks, ready to be executed)
├── protoC_script.py (Script for updating the protobuf files and installing dependencies)    
├── requirements.txt (Packages needed)    
└── requirements_train.txt (Additional Packages needed for training sklearn and pytorch)
```
___
## Setup

### Prerequesite:
1. [Install](https://www.youtube.com/watch?v=dZh_ps8gKgs) protoC
* Ubuntu user can use APT
* Windows user might dowload the latest releas from the [Reposetory](https://github.com/protocolbuffers/protobuf/releases) and add it to the path
2. Install Python (3.9 is required)
* Ubuntu user can use APT, but it should be preinstalled
* Windows user are on their own
3. Install an IDE of your choice
4. **Optional** [Create](https://docs.python.org/3/tutorial/venv.html) a python virtual environment for this project

### Install the API
1. Run ``` python3 protoC_script.py [--dev] ```
2. Fix errors that arise during script execution (ignore tensorflow warnings)
3. You are ready to go

___

## How to Use the API
### Import
In the beginning of your scripts import ```from owlracer.env import Env```
* **Env** is the API that connects to the server. It implements the interface used by OpenAi Gym.

For an easy example have a look at samples/drive_circles.py, which can drive on map 0.
For further information on the functions consult [OpenAi Gym](https://github.com/openai/gym/blob/master/docs/creating-environments.md).
The service can be used with different examples from OpenAi Gym in the near future.

### Run

1. Start the owlracer C# server (**Optional** Start the C# client in spectator mode )
2. Start your script, the API will connect to the server
3. Have fun tweaking your scripts

---

## Train models
In the directory `examples/train` you can find many examples how you can train models with pytorch, sklearn and 
hyperoptimization. Root directory: owlracer-ai-opensource-client-python.
### train_sklearn.py
Training data is specified by `--data`. Experiment name must be provided with `--experiment`.

During the training metrics, artifacts and models will be logged with mlflow. You can add a connection for mlflow server via the option `--server-log`. Parameters for the server must be specified in `.env`-File in root directory. See .envtemplate for required parameters. The `MLFLOW_S3_BUCKET`-variable is optional and specifies the artifact storage location. Note that the artifact storage location cannot be changed for already existing experiments. If the variable is not specified and the experiment does not exist yet, then the default artifact root from the server will be used.

If the `--server-log` option is left out, logging will happen locally. The results can be found in the `mlruns`-subfolders or alternatively can be retrieved on a local mlflow-Server that can be opened with `mlflow ui` from command line (must be executed in the directory one level above the `mlruns`-folder).


* (Hyper-)parameters need to be set in `examples/train/params.yaml`. 
* Preprocessing can be adapted in `examples/train/owlracer_dataset.py`.
* Use a labelmap such as `examples/train/labelmap.yaml` to map the classes and verify consistency of the stepcommands between different frameworks.
* The models need to be exported to ONNX to be able to be executed with all of the provided clients. Therefore, it is in some cases necessary to adapt the exported onnx outputs to provide compatibility, e.g. `examples/train/train_pytorch.py`

## ONNX Export
necessary output nodes are:

* output_probability
* output_label

## Executing Models

ONNX-models can be executed with `examples/play/simpleML.py`. The model path is specified by `--model`. It is important to note that at the moment, the script can only handle ONNX-models which contain exactly one array of output probabilities and an arbitrary amount of other outputs of type `np.int64`. The script disregards all outputs of type `np.int64` and applies argmax to the array of output probabilities, thus extracting the command which should be executed by the model. This is done in order to ensure compatibility between models resulting from training with different packages, e.g. sklearn, pytorch, etc. The exact way in which this is done should maybe be updated in the future, when more types of models are added.