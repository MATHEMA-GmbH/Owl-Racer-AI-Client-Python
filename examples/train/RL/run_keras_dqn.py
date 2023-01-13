import os
import sys

import numpy as np
from keras.models import load_model

from keras_dqn import get_model

from owlracer import owlParser
from owlracer.services import Command
from owlracer.env import CarEnv


def play_dqn(env: CarEnv, model: str):
    # if not dqn agent passed, build keras model
    # Option 1: load keras model
    dqn = load_model(model)

    # Option 2: build model and load weights
    #dqn = get_model()
    #dqn.load_weights(f"model-me-2a.hdf5")

    action = Command.idle

    print("Playing")
    for i in range(5000):
        observation, reward, terminated, info = env.step(action)
        observation = np.expand_dims(observation, axis=0)
        observation = np.expand_dims(observation, axis=0)

        if terminated:
            env.reset()
        action_probs = dqn.predict_on_batch(observation)
        action = np.argmax(action_probs)


@owlParser
def mainLoop(args):
    args = vars(args)

    if "model" not in args.keys():
        print("error, model not selected")
        sys.exit(1)
    else:
        model_path = args["model"]

    remove = []
    args.pop("model")
    for key in args.keys():
        if args[key] is None:
            remove.append(key)

    for item in remove:
        args.pop(item)
    print(args)

    env = CarEnv(**args)
    env.action_space.seed(42)

    play_dqn(env=env, model=model_path)

    #env.close()


if __name__ == '__main__':

    mainLoop()



