import random
import string
from matlabs.owlracer import core_pb2
from grpcClient import core_pb2_grpc as core
from grpcClient import retry_after_exception
import grpc
import numpy as np
import logging



# create LOGGER
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

EMPTY = core.google_dot_protobuf_dot_empty__pb2.Empty()

class GrpcConnector:

    def __init__(self, channel=None, ip='localhost', port='6003', max_retries=3, timeout=5):
        """
        Constructor of the GrpcConnector(), which is the facade for the grpc related cals.
        @param channel: if a channel already exists else None
        @param ip: ip of the server
        @param port: port of the server
        @param max_retries: number of retries should a disconnect or grpc error occur, set for class globally
        @param timeout: number of second to wait while checking stable connection, set for class globally
        """

        self._service = None
        self._channel = channel
        self.ip = ip
        self.port = port
        self.max_retries = max_retries
        self.timeout = timeout

        # Flag for stable connection
        self.ready = False

        LOGGER.info('Created Connector')

    def __del__(self):
        """
        Deconstructor that closes the grpc channel
        """
        self.close_channel()

        LOGGER.info('Closed Connector')

    @retry_after_exception
    def open_channel(self, ip=None, port=None):
        """
        opens grpc channel to the server
        @param ip: ip of the server
        @param port: port of the server
        """

        if self._channel is not None:
            self._channel.close()
            LOGGER.info('Closed existing Connector')

        ip = ip or self.ip
        port = port or self.port
        self._channel = grpc.insecure_channel(ip + ':' + port)

        # check if channel is ready
        grpc.channel_ready_future(self._channel).result(self.timeout)
        self.ready = True


    @retry_after_exception
    def close_channel(self):
        """
        Closes grpc channel with the server
        """

        self._channel.close()

    @retry_after_exception
    def create_service(self):
        """
        Creates grpc service based on the protobuf file
        """

        self._service = core.GrpcCoreServiceStub(channel=self._channel)

    @retry_after_exception
    def create_session(self, trackNr=2, gameTime=50.0, sessionName=""):
        """
        Creates new game session on the server
        @param trackNr: nr of the track to load
        @param gameTime: time calculation multiplier
        @param sessionName: name of the session on the server
        @return: current session state
        """

        # generate random name if no session name is given
        if sessionName == "":
            sessionName = "".join(random.choices(string.ascii_letters + string.digits, k=4))

        sessionData = core_pb2.CreateSessionData(**{'gameTimeSetting': gameTime, 'name': sessionName,
                                                    'trackNumber': trackNr})
        session: core_pb2.SessionData = self._service.CreateSession(sessionData)

        LOGGER.info('Created Session {}'.format(str(session.id)))
        return session

    @retry_after_exception
    def get_session(self, sessionID):
        """
        Get recent session data
        @param sessionID: ID of the session to update
        @return: recent sessionData
        """

        session: core_pb2.SessionData = self._service.GetSession(core_pb2.GuidData(guidString=sessionID))

        LOGGER.info('Created Session {}'.format(str(session.id)))

        return session

    @retry_after_exception
    def update_sessions(self):
        """
        Update all session running on the server
        @return: dict with all sessions and cars
        """

        sessions = self._service.GetSessionIds(EMPTY)
        sessions = np.array([str(ID.guidString) for ID in sessions.guids])
        np.expand_dims(sessions, 1)
        allCarsInSession = {}

        for sessionID in sessions:
            cars = self._service.GetCarIds(core_pb2.GuidData(guidString=sessionID))
            cars = np.array([str(ID.guidString) for ID in cars.guids])
            allCarsInSession.update({sessionID: cars})
            LOGGER.info('Updated Cars in Session {}'.format(sessionID))

        return allCarsInSession

    @retry_after_exception
    def create_car(self, sessionID, carName="", carColor="#008800", ):
        """
        Creates new RaceCar on the server
        @param sessionID: ID of the session to create RaceCar
        @param carName: name of the car
        @param carColor: color of the car
        @return: recent RaceCarData
        """

        if carName == "":
            carName = "".join(random.choices(string.ascii_letters + string.digits, k=4))

        newRaceData: core_pb2.RaceCarData = self._service.CreateCar(
            core_pb2.CreateCarData(sessionId=core_pb2.GuidData(guidString=sessionID), maxVelocity=0.5,
                                   acceleration=0.05, name=carName, color=carColor))

        LOGGER.info('Created Car {} in Session {} Successfully'.format(newRaceData.id, sessionID))

        return newRaceData

    @retry_after_exception
    def destroy_car(self, carID):
        """
        Destroys RaceCar on the server
        @param carID: ID of the car do destroy
        """

        self._service.DestroyCar(core_pb2.GuidData(guidString=carID))
        LOGGER.info('Destroied Car {} Successfully'.format(carID))

    @retry_after_exception
    def get_car_data(self, carID):
        """
       Get data of Car given its ID from the server without committing a move
       @param carID: GuidData ID for the car
       @return: RaceCarData
       """
        usedCarID = core_pb2.GuidData(guidString=carID)

        result: core_pb2.RaceCarData = self._service.GetCarData(usedCarID)
        LOGGER.info('Requested CarData {} Successfully'.format(carID))

        return result

    @retry_after_exception
    def step(self, action, car_id):
        """
        Commits action data to the server
        @param action: action data
        @param car_id: id of the car to commit the action
        @return: recent RaceCarData
        """
        stepResult: core_pb2.RaceCarData = self._service.Step(
            core_pb2.StepData(**{'carId': core_pb2.GuidData(guidString=car_id), 'command': action}))

        LOGGER.info('Step command for car {} Successfully'.format(car_id))

        return stepResult

    @retry_after_exception
    def reset(self, car_id):
        """
        Resets the RaceCar to the server defined respawn
        @param car_id: id of the RaceCar to reset
        @return: recent RaceCarData
        """

        self._service.Reset(core_pb2.GuidData(guidString=car_id))
        stepResult: core_pb2.RaceCarData = self._service.Step(
            core_pb2.StepData(**{'carId': core_pb2.GuidData(guidString=car_id), 'command': 0}))

        LOGGER.info('Reset of Car {} Successfully'.format(car_id))

        return stepResult


