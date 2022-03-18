"""
Environment using gym interface, handling the grpc conection and car creation.
"""

import gym
from owlracer.grpcClient.grpcConnector import GrpcConnector
from grpcClient.converter import racecardata_to_car, sessiondata_to_session
import logging

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Env(gym.Env):
    """Environment class using openAI gym interface. This class connects
    the client with the grpc _service. Each object keeps their own connection
    and car after init.
    """

    def __init__(self, channel=None, ip='localhost', port='6003', spectator=False, carName="Default", carColor="#008800", sessionName="",
                 gameTime = 50, gameTrack=2, session=None):
        """Environment class using openAI gym interface. This class connects
        the client with the grpc _service. Each object keeps their own connection
        and car after init.

        @param channel: already opened grpc channel
        @param ip: ip to connect to
        @param port: port of the server
        @param spectator: without interaction
        @param session: session ID to join, if "" it will join first session. If None defined it will create a new session
        """

        self.spectator = spectator
        self.allCarsInSession = {}
        self.Connector = GrpcConnector(channel=channel, ip=ip, port=port)
        self.Connector.open_channel()
        self.Connector.create_service()
        self.updateAllSessions()
        
        # session join logic

        if session is None:
            # create new session if none defined
            self.session = sessiondata_to_session(self.Connector.create_session(trackNr=gameTrack, gameTime=gameTime, sessionName=sessionName))
        else:
            # "" is default value from the arparser to enforce first session if none is defined
            if session == "":
                # connect onto first session in server dict (does not have to be the first session on server)
                sessions = list(self.getSessionIDs())
                if len(sessions) == 0:
                    logger.error("There are no sessions on the server")
                    raise ValueError("There are no sessions on the server")
                session = sessions[0]

            # connect to session
            if session in list(self.allCarsInSession.keys()):
                self.session = sessiondata_to_session(self.Connector.get_session(session))
            else:
                logger.error("Session with ID {} is not available on server".format(session))
                raise ValueError("Session with ID {} is not available on server".format(session))

        # creating new raceCar
        self.car = self.create_car(car_name=carName, car_color=carColor)

        logger.info('Created Env Successfully with Session {} and Car {}'.format(self.car.sessionID, self.car.ID))

    def __del__(self):
        """destroys car and closes connection
        """
        if not self.spectator and self.Connector.ready:
            self.Connector.destroy_car(self.car.ID)
        self.Connector.__del__()
        logger.info('Deleted Env Successfully')

    @property
    def isPaused(self):
        return self.session.isPaused()

    @property
    def isPrerace(self):
        return self.session.isPrerace()

    @property
    def isRace(self):
        return self.session.isRace()

    def step(self, action):
        """
        Sends the action for the next step to the server
        @param action: defined action
        @return: new state of the RaceCar
        """
        return racecardata_to_car(self.Connector.step(action, self.car.ID))

    def reset(self):
        """
        Resets the RaceCar to the Respawn point defined by the server
        @return: new state of the RaceCar
        """
        return racecardata_to_car(self.Connector.reset(self.car.ID))


    def render(self, mode='human'):
        """NOT IMPLEMENTED YET
        """
        logger.info('NOT IMPLEMENTED YET')

    def close(self):
        """Destroys the object and hence closes the connection
        """
        self.__del__()

    def getCarIDs(self, sessionID):
        """
        Get the IDs from all cars on one session
        @return: list of car IDs
        """

        return self.allCarsInSession[sessionID]

    def create_car(self, car_name, car_color):
        """
        Creates new instance of the RaceCar on the server
        @param car_name: name of the RaceCar
        @param car_color: color of the RaceCar
        @return: new state of the RaceCar
        """
        return racecardata_to_car(self.Connector.create_car(self.session.id, carName=car_name, carColor=car_color))

    def getSessionIDs(self):
        """
        Get the IDs of all active sessions
        @return: List of session IDs
        """

        return self.allCarsInSession.keys()

    def getCarData(self, carID=None):
        """
        Get data of Car given its ID from the server without committing a move
        @param carID: GuidData ID for the car
        @return: RaceCarData
        """
        return self.Connector.get_car_data(carID)

    def getSession(self, carID):
        """
        Get the corresponding session ID for the given RaceCar ID
        @param carID: ID of the RaceCar to search
        @return: ID of the session or empty string
        """

        for key, values in self.allCarsInSession.items():
            for value in values:
                if value == carID:
                    return key
        return ""

    def setCarID(self, carID):
        """
        Sets the ID for the car in use by this Class
        @param carID: GuidData ID
        """
        self.carID = carID

    def updateAllSessions(self):
        """
        Method to sync the state of all sessions on the Client with the server
        @return: dict with all sessions and cars
        """

        self.allCarsInSession = self.Connector.update_sessions()
        return self.allCarsInSession

    def updateSession(self):
        """
        Get the recent state of a the joined session for the server
        @return: recent session state
        """

        session = sessiondata_to_session(self.Connector.get_session(self.session.id), self.session)
        self.Connector.session = session
        return session
