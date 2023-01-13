import time

# API import
from owlracer.env import CarEnv
from owlracer import owlParser
from owlracer.services import Command

@owlParser
def main_loop(args):
    args = vars(args)
    args.pop("carName")
    args.pop("carColor")
    args.pop("model")

    env = CarEnv(**args, carColor="#c78422", carName="Drive_Circles_(Py)")
    
    observation, reward, terminated, info = env.step(Command.idle)

    for i in range(1000):

        while env.isPrerace or env.isPaused:
            env.updateSession()
            time.sleep(0.1)

        distance_front = observation[2]
        distance_frontLeft = observation[3]
        if distance_front < 120 or distance_frontLeft < 30:
            observation, reward, terminated, info = env.step(Command.accelerateRight)
        else:
            observation, reward, terminated, info = env.step(Command.accelerate)

        if terminated:
            env.reset()

        # sleep for human
        time.sleep(0.01)
    

if __name__ == '__main__':
    main_loop()
