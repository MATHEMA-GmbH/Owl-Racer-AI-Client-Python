from collections import namedtuple

Score = namedtuple('Score', ['carId', 'carName', 'score'])


class GameSession:


    def __init__(self):
        """
        Class for wrapping the grpc class and provide a python nativ interface to the data.
        Check with the Protobuf file for definition of the parameters.
        """
        self._id = None
        self._name = None
        self._gameTimeSetting = None
        self._trackNumber = None
        self._scores = None
        self._gameTime = None
        self._isGameTimeNegative = None
        self._phase = None

    # check session phase/status

    def isPaused(self):
        return self.phase == 0

    def isPrerace(self):
        return self.phase == 1

    def isRace(self):
        return self.phase == 2

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def gameTimeSetting(self):
        return self._gameTimeSettings

    @gameTimeSetting.setter
    def gameTimeSetting(self, value):
        self._gameTimeSettings = value

    @property
    def trackNumber(self):
        return self._trackNumber

    @trackNumber.setter
    def trackNumber(self, value):
        self._trackNumber = value

    @property
    def scores(self):
        return self._scores

    @scores.setter
    def scores(self, values):
        self._scores = [Score(value.carId.guidString, value.carName, value.score) for value in values]

    @property
    def gameTime(self):
        return self._gameTime

    @gameTime.setter
    def gameTime(self, value):
        self._gameTime = value

    @property
    def gameTime(self):
        return self._gameTime

    @gameTime.setter
    def gameTime(self, value):
        self._gameTime = value

    @property
    def isGameTimeNegative(self):
        return self._isGameTimeNegative

    @isGameTimeNegative.setter
    def isGameTimeNegative(self, value):
        self._isGameTimeNegative = value

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value
