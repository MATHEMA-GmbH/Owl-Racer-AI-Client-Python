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
In the directory `examples/train` you can find examples on how you can train models with pytorch and sklearn. Root directory: owlracer-ai-opensource-client-python.
### Parameters for the train scripts
* Training data is specified by `--data`. 
* Experiment name must be provided with `--experiment`.
* During the training metrics, artifacts and models will be logged with mlflow. You can add a connection for mlflow server via the option `--server-log`. Parameters for the server must be specified in `.env`-File in root directory. See .envtemplate for required parameters. The `MLFLOW_S3_BUCKET`-variable is optional and specifies the artifact storage location. Note that the artifact storage location cannot be changed for already existing experiments. If the variable is not specified and the experiment does not exist yet, then the default artifact root from the server will be used.
* If the `--server-log` option is left out, logging will happen locally. The results can be found in the `mlruns`-subfolders or alternatively can be retrieved on a local mlflow-Server that can be opened with `mlflow ui` from command line (must be executed in the directory one level above the `mlruns`-folder).

### Specifying used features for training

At the moment, the following features occur in the training data: 
* IsCrashed 
* MaxVelocity 
* Position.X 
* Position.Y 
* Checkpoint 
* Rotation 
* ScoreStep 
* ScoreOverall 
* ScoreChange 
* Ticks 
* Velocity 
* Distance.Front 
* Distance.FrontLeft 
* Distance.FrontRight 
* Distance.Left 
* Distance.Right 
* WrongDirection 

In `examples/train/model_config.yaml` under `features.used_features`, one can specify which of the aforementioned features shall be used in the trainig.

### Specifying the command set

The following commands can be used by the owlracer car:
* Idle = 0
* Accelerate = 1
* Decelerate = 2
* AccelerateLeft = 3
* AccelerateRight = 4
* TurnLeft = 5
* TurnRight = 6

Sometimes not all of the available commands shall be used by the model. In this case there are two possibilities on how to handle the unused commands, which can also be used in conjunction:
1. In `examples/train/model_config.yaml` under `change_commands.drop_commands` one can specify a list of commands that should be dropped from the training data entirely, in the sense of removing rows which correspond to those commands.
2. In `examples/train/model_config.yaml` under `change_commands.replace_commands.commands` one can specify a dictionary on how to replace a specific command with another. If `change_commands.replace_commands.before_training` is set to `True`, this will happen during training (as a preprocessing step), otherwise the commands will be replaced during execution of the model.

Note that during training, a `labelmap.yaml`-file is automatically generated and logged via mlflow.

### Adjusting training parameters and preprocessing
* (Hyper-)parameters need to be set in `examples/train/params.yaml`. 
* Preprocessing can be adapted in `examples/train/owlracer_dataset.py`.
* The models need to be exported to ONNX to be able to be executed with all of the provided clients. Therefore, it is in some cases necessary to adapt the exported onnx outputs to provide compatibility, e.g. `examples/train/train_pytorch.py`. For more info see the next two chapters.

## ONNX Export
Exactly one array of output probabilities is necessary. An arbitrary amount of other outputs of type `np.int64` are possible, but will be disregarded in the execution of the model. For more info, see the next chapter.

## Executing Models

ONNX-models can be executed with `examples/play/simpleML.py`. The model path is specified by `--model`. In this model path, at least three files are mandatory:
* The onnx-file of the model;
* The `model_config.yaml` with which the model was trained;
* The `labelmap.yaml` which was automatically generated during training of the model.

In the case of pytorch-Models, there is an additional file needed: During training with pytorch, a file with the same name as the run ID is generated. This file is also needed when executing the model. For an example, see the folder structure of `trainedModels/NN/`. 

Note that all of the files mentioned above are logged via mlflow.

It is important to note that at the moment, the script can only handle ONNX-models which contain exactly one array of output probabilities and an arbitrary amount of other outputs of type `np.int64`. The script disregards all outputs of type `np.int64`.

Using this array of probabilities, the command which shall be executed by the model is then extracted as specified by `--actionChoice`; There are two possibilities: `argmax` and `probabilities`. `argmax` is the default and means that the command with the highest probability will be chosen. `probabilities` means that the command is chosen randomly where the probability distribution is given by the array of output probabilities. 

This is done in order to ensure compatibility between models resulting from training with different packages, e.g. sklearn, pytorch, etc. The exact way in which this is done should maybe be updated in the future, when more types of models are added.
