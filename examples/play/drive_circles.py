import time

# API import
from owlracer.env import Env as Owlracer_Env
from owlracer import owlParser
from owlracer.services import Command

@owlParser
def main_loop(args):
  
    env = Owlracer_Env(ip=args.ip, port=args.port, spectator=args.spectator, session="", gameTrack=1,
                       carColor="#c78422", carName="Drive_Circles_(Py)")
    
    step_result = env.step(0)

    for i in range(1000):


        while env.isPrerace or env.isPaused:
            env.updateSession()
            time.sleep(0.1)

        if step_result.distance.front < 120 or step_result.distance.frontLeft < 30:
            step_result = env.step(Command.spinRight)
        else:
            step_result = env.step(Command.accelerate)

        if step_result.isCrashed:
            env.reset()

        print("Car Pos: {} {}, Vel: {} forward distance {}"
              .format(step_result.position.x, step_result.position.y, step_result.velocity, step_result.distance.front))

        # sleep for human
        time.sleep(0.01)

    print(step_result)
    print("My car {}".format(step_result.ID))
    

if __name__ == '__main__':
    main_loop()
