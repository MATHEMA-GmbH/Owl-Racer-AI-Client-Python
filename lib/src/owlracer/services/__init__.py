"""
Service Environment for usage by the client. This is for convinient use. Contains service class for hadling communication and commands for easy reading
"""

"""Helper variables for commanding the car """
from enum import IntEnum


class Command(IntEnum):

    idle = 0
    accelerate = 1
    decelerate = 2
    accelerateLeft = 3
    accelerateRight = 4
    turnLeft = 5
    turnRight = 6
