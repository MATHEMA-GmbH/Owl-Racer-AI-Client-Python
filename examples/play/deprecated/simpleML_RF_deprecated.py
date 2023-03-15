import time
import numpy as np
import os

# API import
from owlracer import owlParser
from owlracer.env import CarEnv
from owlracer.services import Command
import onnx
import onnxruntime


class OwlRacerEnv(CarEnv):
    """
    Wraps the Owlracer Env and returns the choosen variabels
    Args:
        service (class): OwlracerAPI env
    """

    def __init__(self, session):

        super().__init__(carColor="#a8a60d", carName="RandomForest_deprecated(Py)", session=session) #gameTime=20
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

        #shape of (x,1)
        step = {'Velocity':             np.array([[np.float32(observation[1])]]),
                'Distance_Front':       np.array([[np.int64(observation[2])]]),
                'Distance_FrontLeft':   np.array([[np.int64(observation[3])]]),
                'Distance_FrontRight':  np.array([[np.int64(observation[4])]]),
                'Distance_Left':        np.array([[np.int64(observation[5])]]),
                'Distance_Right':       np.array([[np.int64(observation[6])]])}

        return step


@owlParser
def mainLoop(args):
    model_name = "../../../trainedModels/deprecated-sklearn/RF.onnx"
    this_dir = os.path.dirname(__file__)
    model_name = os.path.abspath(os.path.join(this_dir, model_name))
    model = onnx.load(model_name)

    # Check the model
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)
    else:
        print('The model is valid!')

    session = onnxruntime.InferenceSession(model_name)

    # Start owlracer Env
    env = OwlRacerEnv(session=args.session)

    step = env.step(Command.idle)

    # play the game forever

    while True:

        # waiting for game start
        while env.isPrerace or env.isPaused:
            env.updateSession()
            time.sleep(0.1)

        action = session.run(None, step)[0][0]
        step = env.step(action)


if __name__ == '__main__':
    mainLoop()
