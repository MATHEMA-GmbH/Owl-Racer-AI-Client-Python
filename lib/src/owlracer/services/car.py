
import pandas as pd
from collections import namedtuple

Position = namedtuple('Position', ['x', 'y'])
Distances = namedtuple('Distances', ["front", "frontLeft", "frontRight", "left", "right", "maxViewDistance"])

class RaceCar:

    def __init__(self, ID=None, sessionID=None, position=pd.DataFrame({"x":[0],"y":[0]}), rotation=0.0, maxVelocity=0.0,
                 acceleration=0.0, velocity=0.0, isCrashed=False, scoreStep=0, scoreOverall=0, ticks=0,
                 distance=pd.DataFrame({"front":[0], "frontLeft":[0], "frontRight":[0], "left":[0], "right":[0],
                                        "maxViewDistance":[0]}),
                 checkPoint=0, lastStepCommand=0, name="", numRounds=0, numCrashes=0, wrongDirection=False, scoreChange=0):
        """
        Class for wrapping the grpc class and provide a python nativ interface to the data.
        Check with the Protobuf file for definition of the parameters.
        @param ID:
        @param sessionID:
        @param position:
        @param rotation:
        @param maxVelocity:
        @param acceleration:
        @param velocity:
        @param isCrashed:
        @param scoreStep:
        @param scoreOverall:
        @param ticks:
        @param distance:
        @param checkPoint:
        @param lastStepCommand:
        """
        self.ID = ID
        self.sessionID = sessionID
        self.position = position
        self.rotation = rotation
        self.maxVelocity = maxVelocity
        self.acceleration = acceleration
        self.velocity = velocity
        self.isCrashed = isCrashed
        self.scoreStep = scoreStep
        self.scoreOverall = scoreOverall
        self.ticks = ticks
        self.distance = distance
        self.checkPoint = checkPoint
        self.lastStepCommand = lastStepCommand
        self.name = name
        self.numRounds = numRounds
        self.numCrashed = numCrashes
        self.wrongDirection = wrongDirection
        self.scoreChange = scoreChange

    @property
    def ID(self):
        return self._ID

    @ID.setter
    def ID(self, value):
        self._ID = value

    @property
    def sessionID(self):
        return self._sessionID

    @sessionID.setter
    def sessionID(self, value):
        self._sessionID = value

    @property
    def position(self):
        return Position(float(self._position.x), float(self._position.y))

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = value

    @property
    def maxVelocity(self):
        return self._maxVelocity

    @maxVelocity.setter
    def maxVelocity(self, value):
        self._maxVelocity = value

    @property
    def acceleration(self):
        return self._acceleration

    @acceleration.setter
    def acceleration(self, value):
        self._acceleration = value

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        self._velocity = value

    @property
    def isCrashed(self):
        return self._isCrashed

    @isCrashed.setter
    def isCrashed(self, value):
        self._isCrashed = value

    @property
    def scoreStep(self):
        return self._scoreStep

    @scoreStep.setter
    def scoreStep(self, value):
        self._scoreStep = value

    @property
    def scoreOverall(self):
        return self._scoreOverall

    @scoreOverall.setter
    def scoreOverall(self, value):
        self._scoreOverall = value

    @property
    def ticks(self):
        return self._ticks

    @ticks.setter
    def ticks(self, value):
        self._ticks = value

    @property
    def distance(self):
        return Distances(float(self._distance.front), float(self._distance.frontLeft), float(self._distance.frontRight),
                         float(self._distance.left), float(self._distance.right), float(self._distance.maxViewDistance))

    @distance.setter
    def distance(self, value):
        self._distance = value

    @property
    def checkPoint(self):
        return self._checkPoint

    @checkPoint.setter
    def checkPoint(self, value):
        self._checkPoint = value

    @property
    def lastStepCommand(self):
        return self._lastStepCommand

    @lastStepCommand.setter
    def lastStepCommand(self, value):
        self._lastStepCommand = value

    @property
    def scoreChange(self):
        return self._scoreChange

    @scoreChange.setter
    def scoreChange(self, value):
        self._scoreChange = value

    @property
    def wrongDirection(self):
        return self._wrongDirection

    @wrongDirection.setter
    def wrongDirection(self, value):
        self._wrongDirection = value