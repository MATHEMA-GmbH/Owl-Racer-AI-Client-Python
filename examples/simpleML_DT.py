import time
import numpy as np
import os

# API import
from owlracer import owlParser
from owlracer.env import Env as Owlracer_Env
from owlracer.services import Command
import onnx
import onnxruntime

class OwlRacerEnv(Owlracer_Env):
    """
    Wraps the Owlracer Env and returns the choosen variabels
    Args:
        service (class): OwlracerAPI env
    """

    def __init__(self, session):

        super().__init__(carColor="#a8a60d", carName="Decision Tree (Py)", session=session) #gameTime=20
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
        step_result = super().step(action)

        # shape of (x,1)
        step = {'Velocity':             np.array([[np.float32(step_result.velocity)]]),
                'Distance_Front':       np.array([[np.int64(step_result.distance.front)]]),
                'Distance_FrontLeft':   np.array([[np.int64(step_result.distance.frontLeft)]]),
                'Distance_FrontRight':  np.array([[np.int64(step_result.distance.frontRight)]]),
                'Distance_Left':        np.array([[np.int64(step_result.distance.left)]]),
                'Distance_Right':       np.array([[np.int64(step_result.distance.right)]])}

        self.posDiffSq = max((self.posX - step_result.position.x) ** 2, (self.posY - step_result.position.y) ** 2)
        self.sameCommand = self.lastCommand == step_result.lastStepCommand

        self.posX = step_result.position.x
        self.posY = step_result.position.y
        self.lastCommand = step_result.lastStepCommand

        return [step, step_result]

    def reset(self):
        """
        Resets the RaceCar to the server Respawn point
        Returns:
            tuple: next state
        """

        step_result = super().reset()

        # shape of (x,1)
        step = {'Velocity': np.array([[np.float32(step_result.velocity)]]),
                'Distance_Front': np.array([[np.int64(step_result.distance.front)]]),
                'Distance_FrontLeft': np.array([[np.int64(step_result.distance.frontLeft)]]),
                'Distance_FrontRight': np.array([[np.int64(step_result.distance.frontRight)]]),
                'Distance_Left': np.array([[np.int64(step_result.distance.left)]]),
                'Distance_Right': np.array([[np.int64(step_result.distance.right)]])}

        return [step, step_result]

    def is_moving(self):

        if self.posDiffSq < 1 and self.car.velocity < 0.1:
            return False

        return True

@owlParser
def mainLoop(args):
    model_name = "../trainedModels/DT.onnx"
    this_dir = os.path.dirname(__file__)
    model_name = os.path.join(this_dir, model_name)
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

    step, step_result = env.step(Command.idle)

    # play the game forever

    while(True):

        # waiting for game start
        while env.isPrerace or env.isPaused:
            env.updateSession()
            time.sleep(0.1)

        start_inf = time.time()
        action = session.run(None, step)[0][0]
        duration_inf = time.time() - start_inf

        start_step = time.time()
        last_tick = step_result.ticks
        step, step_result = env.step(action)
        duration_step = time.time() - start_step

        # check if stuck
        if not env.is_moving():
            env.step(Command.accelerate)
        #print("Car Pos: {} {}, Vel: {} forward distance {}".format(step_result.position.x, step_result.position.y,
        #                                                           step_result.velocity, step_result.distance.front))
        #print("Time for executing inf {} or in ticks {}, and step {}".format(duration_inf, step_result.ticks - last_tick, duration_step))
        #print(step_result)


if __name__ == '__main__':
    mainLoop()
