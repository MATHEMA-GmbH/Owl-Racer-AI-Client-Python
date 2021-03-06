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
├── data (Data captured for subervised learning)   
├── jupyter (example Pythone Notebooks)  
├── lib (Package for building the python API)     
├── pytorch (example for using pytorch)     
├── protoC_script.py (Script for updating the protobuf files and installing dependencies)    
├── requirements.txt (Packages needed)    
└── exsamples (Easy to follow and extend)   
```
___
## Setup

### Prerequesite:
1. [Install](https://www.youtube.com/watch?v=dZh_ps8gKgs) protoC
* Ubuntu user can use APT
* Windows user might dowload the latest releas from the [Reposetory](https://github.com/protocolbuffers/protobuf/releases) and add it to the path
2. Install Python > 3.6
* Ubuntu user can use APT, but it should be preinstalled
* Windows user are on their own
3. Install an IDE of your choice
4. **Optional** [Create](https://docs.python.org/3/tutorial/venv.html) a python virtual environment for this project

### Install the API
1. Run ``` python3 protoC_script.py ```
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
