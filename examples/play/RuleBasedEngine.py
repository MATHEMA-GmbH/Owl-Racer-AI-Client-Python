import time
from owlracer.env import Env as Owlracer_Env
from owlracer import owlParser


def calculate_action(step_result, list):
    distance_right = step_result.distance.right
    distance_front_right = step_result.distance.frontRight
    distance_left = step_result.distance.left
    distance_front_left = step_result.distance.frontLeft

    if list["fixed_left"] > 0:
        list["fixed_left"] = list["fixed_left"]-1
        if list["fixed_left"] > 30:
            return 2
        else:
            return 3

    elif list["fixed_right"] > 0:
        list["fixed_right"] = list["fixed_right"]-1
        if list["fixed_right"] > 30:
            return 2
        return 4

    elif distance_left > 200 and list["fixed_left"] == 0:
        list["fixed_left"] = 80
        print("distance left big!")
        return 2

    elif distance_right > 200 and list["fixed_right"] == 0:
        list["fixed_right"] = 80
        print("distance left big!")
        return 2

    else:
        if distance_front_left == 0:
            ratio = distance_front_right/(distance_front_left + 0.00001)
        else:
            ratio = float(distance_front_right)/distance_front_left

        if step_result.distance.front >= 50:
            if ratio < 1:
                return 3
            elif ratio > 1:
                return 4
            else:
                return 1

        else:
            if ratio < 1:
                return 5
            elif ratio > 1:
                return 6
            else:
                return 2

@owlParser
def main_loop(args):
    print(args.session)
    env = Owlracer_Env(ip=args.ip, port=args.port, spectator=args.spectator, session=args.session, carName="Rule-based (Py)", carColor="#07f036")
    step_result = env.step(0)


    list ={
        "fixed_left": 0,
        "fixed_right": 0
    }

    while True:

        # waiting for game to start
        while env.isPrerace or env.isPaused:
            env.updateSession()
            time.sleep(0.1)

        action = calculate_action(step_result, list)

        step_result = env.step(action)

        print("Car Left/right: {} {}, Vel: {} forward distance {}".format(step_result.distance.left, step_result.distance.right,
                                                                   step_result.velocity, step_result.distance.front))
        # sleep for human
        time.sleep(0.01)


if __name__ == '__main__':
    main_loop()
