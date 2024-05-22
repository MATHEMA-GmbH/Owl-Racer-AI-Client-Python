import time
import numpy as np
import os
import sys

import yaml

# API import
from owlracer import owlParser
from owlracer.env import CarEnv
from owlracer.services import Command
import onnx
import onnxruntime

def get_action(action):
    # returns output probabilities from model
    if len(action) > 1:
        for item in action:
            if type(item[0]) is np.int64:
                pass
            elif len(item[0]) > 1:
                return item
        return [0]
    else:
        return action[0]


def get_transform_observation(observation, features):
    return_list = []
    for feature in features["used_features"]:
        # get values of observed features;
        # only features which where used for training according to model_config.yaml are considered.
        feature_value = observation[feature]
        if feature in features["normalization_constants"].keys():
            # normalization according to model_config.yaml
            normalization_constant = features["normalization_constants"][feature]
            feature_value = feature_value*normalization_constant
        return_list.append(feature_value)
    return {'input': [return_list]}


@owlParser
def mainLoop(args):
    args = vars(args) # returns dict with {'var1': val1, 'var2', val2, ...}

    if "model" not in args.keys():
        print("error, model not selected")
        sys.exit(1)

    this_dir = os.path.dirname(__file__)

    print(args["model"].replace("\\", "/"))

    model_name = os.path.abspath(os.path.join(this_dir, args["model"].replace("\\", "/")))
    model = onnx.load(model_name)

    remove = []
    args.pop("model")
    for key in args.keys():
        if args[key] is None:
            remove.append(key)

    for item in remove:
        args.pop(item)
    print(args)

    label_map_path = os.path.join(os.path.dirname(model_name), "labelmap.yaml")
    config_path = os.path.join(os.path.dirname(model_name), "model_config.yaml")
    idx2class = yaml.safe_load(open(label_map_path))["idx2class"]
    model_config = yaml.safe_load(open(config_path))
    replace_commands = model_config["change_commands"]["replace_commands"]

    # Check the model
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)
    else:
        print('The model is valid!')

    session = onnxruntime.InferenceSession(model_name)

    #Start owlracer Env
    env = CarEnv(**args)
    observation, reward, terminated, info = env.step(Command.idle)
    step = get_transform_observation(observation, model_config["features"])

    #play the game forever
    while True:

        # waiting for game start
        while env.isPrerace or env.isPaused:
            env.updateSession()
            time.sleep(0.1)

        action = session.run(None, step)

        action = get_action(action) # yields output probabilities for actions/commands

        print(f"action probabilities: {action}")

        # now the actual used action/command is chosen
        if args["actionChoice"] == "argmax":
            action = np.argmax(action)
        elif args["actionChoice"] == "probabilities":
            # Choose action randomly according to the action probabilities
            action = np.random.choice([i for i in range(len(action[0]))], p=action[0])
        else:
            print("error, actionChoice not valid. Valid parameters for actionChoice are \"argmax\",\"probabilities\".")
            sys.exit(1)

        print(f"chosen action: {action}")

        action = idx2class[action]

        if ((replace_commands["commands"] != None) and (action in replace_commands["commands"].keys()) and (replace_commands["before_training"] == False)):
            action_before_change = action
            action = replace_commands["commands"][action]
            print(f"action {action_before_change} replaced with {action}")

        observation, reward, terminated, info = env.step(action)
        step = get_transform_observation(observation, model_config["features"])

        # # check if stuck (model should learn this by itself)
        # if not env.is_moving():
        #     env.step(Command.accelerate)

if __name__ == '__main__':
    mainLoop()
