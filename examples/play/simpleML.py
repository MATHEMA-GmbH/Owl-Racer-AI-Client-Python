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
    if len(action) > 1:
        for item in action:
            if type(item[0]) is np.int64:
                pass
            elif len(item[0]) > 1:
                return item
        return [0]
    else:
        return action[0]


def get_transform_observation(observation):
    # do normalization; must be the same as in change_datatype of OwlracerPreprocessor-class
    # observation-format: [ScoreChange, Distance.Front, ..., Distance.Right, WrongDirection ]
    return_list = []
    return_list.append(observation[0]/10)
    for i in range(5):
        return_list.append(observation[i+1]/1000)
    return_list.append(observation[6])
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

    idx2class = yaml.safe_load(open(label_map_path))["classes"]

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
    step = get_transform_observation(observation)

    #play the game forever
    while True:

        # waiting for game start
        while env.isPrerace or env.isPaused:
            env.updateSession()
            time.sleep(0.1)

        action = session.run(None, step)
        action = get_action(action)

        print(f"action probabilities: {action}")

        if args["actionChoice"] == "argmax":
            action = np.argmax(action)
        elif args["actionChoice"] == "probabilities":
            action = np.random.choice([i for i in range(len(action[0]))], p=action[0])
        else:
            print("error, actionChoice not valid")
            sys.exit(1)

        action = idx2class[action]

        observation, reward, terminated, info = env.step(action)
        step = get_transform_observation(observation)

        # # check if stuck (model should learn this by itself)
        # if not env.is_moving():
        #     env.step(Command.accelerate)

if __name__ == '__main__':
    mainLoop()
