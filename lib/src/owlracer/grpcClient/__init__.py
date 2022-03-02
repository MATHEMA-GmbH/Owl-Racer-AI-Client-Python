"""
Service methods for connecting with the grpc Server from the client. Generated from Protofile for direct connection.
"""

import logging
import grpc
logging.basicConfig(filename='./example.log', filemode='w', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

def retry_after_exception(func):
    """
    A decorator to retry a given function for a given number of retries, if a expected exception is raised
    """

    def wrapper(self, *args, **kwargs):
        # we want to execute the function at least once
        retries = self.max_retries + 1
        result = None
        while retries > 0:
            try:
                result = func(self, *args, **kwargs)
            except grpc.RpcError as error:
                # if a grpc exception occurs, we retry
                LOGGER.error("%s occurred. Restarting %s (%d/%d)", str(error.code()), func.__name__,
                             (self.max_retries + 2) - retries, (self.max_retries + 1))
                retries -= 1
                if retries > 0:
                    continue
                raise error

            except grpc.FutureTimeoutError:
                # if a error in the server connection occurs, we retry
                LOGGER.error("Connection error occurred. Restarting %s (%d/%d)", func.__name__,
                             (self.max_retries + 2) - retries, (self.max_retries + 1))
                # Reset half opened grpc channel
                # self._channel = None
                retries -= 1
                if retries > 0:
                    continue
                raise

            else:
                # no exception was thrown, so we don't need to retry
                retries = 0
                LOGGER.info('Successfully executed {}'.format(func.__name__))
        return result

    return wrapper
