from matlabs.owlracer import core_pb2
from services.car import RaceCar
from services.session import GameSession
import pandas as pd


def racecardata_to_car(newCarData: core_pb2.RaceCarData, car: RaceCar = None):
    """
    Static method to convert the grpc data into a python nativ object.
    @param newCarData: grpc RaceCarData defined via Protobuf file
    @param car: object of an existing RaceCar
    @return: new or changed RaceCar object
    """

    car = car or RaceCar()

    car.ID = newCarData.id.guidString
    car.sessionID = newCarData.sessionId
    car.position = pd.DataFrame({"x": [newCarData.position.x], "y": [newCarData.position.y]})
    car.rotation = newCarData.rotation
    car.maxVelocity = newCarData.maxVelocity
    car.acceleration = newCarData.acceleration
    car.velocity = newCarData.velocity
    car.isCrashed = newCarData.isCrashed
    car.scoreStep = newCarData.scoreStep
    car.scoreOverall = newCarData.scoreOverall
    car.ticks = newCarData.ticks
    car.distance = pd.DataFrame({"front": [newCarData.distance.front], "frontLeft": [newCarData.distance.frontLeft],
                                 "frontRight": [newCarData.distance.frontRight], "left": [newCarData.distance.left],
                                 "right": [newCarData.distance.right],
                                 "maxViewDistance": [newCarData.distance.maxViewDistance]})
    car.checkPoint = newCarData.checkPoint
    car.lastStepCommand = newCarData.lastStepCommand
    car.name = newCarData.name
    car.numRounds = newCarData.numRounds
    car.numCrashes = newCarData.numCrashes
    car.wrongDirection = newCarData.wrongDirection
    car.scoreChange = newCarData.scoreChange

    return car

def sessiondata_to_session(sessionData: core_pb2.SessionData, session: GameSession = None):
    """
    Static method to convert the grpc data into a python nativ object.
    @param sessionData: grpc SessionData defined via Protobuf file
    @param session: object of an existing Session
    @return: new or changed Session object
    """

    if session is not None:
        session = session
    else:
        session = GameSession()

    session.id = sessionData.id.guidString
    session.name = sessionData.name
    session.gameTimeSetting = sessionData.gameTimeSetting
    session.trackNumber = sessionData.trackNumber
    session.scores = sessionData.scores
    session.gameTime = sessionData.gameTime.ToDatetime()
    session.isGameTimeNegative = sessionData.isGameTimeNegative
    session.phase = sessionData.phase

    return session