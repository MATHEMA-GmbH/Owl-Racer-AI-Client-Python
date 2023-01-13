import time
from owlracer.env import CarEnv
from owlracer import owlParser
from owlracer.services import Command


def calculate_action(observation, list):
    distance_right = observation[6]
    distance_front_right = observation[4]
    distance_left = observation[5]
    distance_front_left = observation[3]
    distance_front = observation[2]

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

        if distance_front >= 50:
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
    args = vars(args)
    args.pop("carName")
    args.pop("carColor")
    args.pop("model")

    env = CarEnv(**args, carName="Rule-based (Py)", carColor="#07f036")
    observation, reward, terminated, info = env.step(Command.idle)


    list ={
        "fixed_left": 0,
        "fixed_right": 0
    }

    while True:

        # waiting for game to start
        while env.isPrerace or env.isPaused:
            env.updateSession()
            time.sleep(0.1)

        action = calculate_action(observation, list)

        observation, reward, terminated, info = env.step(action)

        # sleep for human
        time.sleep(0.01)


if __name__ == '__main__':
    main_loop()
