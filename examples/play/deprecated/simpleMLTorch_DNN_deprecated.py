import time
import numpy as np
import os

import yaml

from owlracer import owlParser
from owlracer.env import CarEnv
from owlracer.services import Command
import onnx
import onnxruntime


class OwlRacerEnv(CarEnv):
    """
    Wraps the Owlracer Env and returns the chosen variables
    Args:
        service (class): OwlracerAPI env
    """

    def __init__(self, session):

        super().__init__(carColor="#4f062d", carName="DNN_deprecated(Py)", session=session)
        self.posX = 0
        self.posY = 0
        self.lastCommand = Command.idle
        #just larger than 1
        self.posDiffSq = float('inf')

    def step(self, action):
        """
        Commit Step with given action
        Args:
            action (int): action to performe

        Returns:
            tuple: next state
        """
        observation, reward, terminated, info = super().step(action)

        # shape of (x,1)
        step = {'input': [observation[1:]]}

        return step

@owlParser
def mainLoop(args):

    this_dir = os.path.dirname(__file__)

    model_name = "../../../trainedModels/deprecated-pytorch/DNN.onnx"
    model_name = os.path.abspath(os.path.join(this_dir, model_name))
    model = onnx.load(model_name)

    label_map_path = "../../../trainedModels/deprecated-pytorch/labelmap.yaml"
    label_map_path = os.path.abspath(os.path.join(this_dir, label_map_path))
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
    env = OwlRacerEnv(session=args.session)

    step = env.step(Command.idle)
    print(step)

    #play the game forever
    while (True):

        # waiting for game start
        while env.isPrerace or env.isPaused:
            env.updateSession()
            time.sleep(0.1)

        action = session.run(None, step)
        if len(action) > 1:
            action = action[1]
        else:
            action = action[0]
        action = idx2class[np.argmax(action)]
        step = env.step(action)


if __name__ == '__main__':
    mainLoop()
