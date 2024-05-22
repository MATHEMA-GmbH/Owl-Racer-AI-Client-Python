
import sys
import os
import site

packageSite = site.getsitepackages()[-1]
packageSite = os.path.join(packageSite,"owlracer")
sys.path.insert(0, packageSite)

#Setting Global Logger
import logging
logging.basicConfig(filename='./example.log', filemode='w', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Argparse
import argparse

PARSER = argparse.ArgumentParser(description='Commandline Interface for using Owlracer API')
PARSER.add_argument('--ip', type=str, default='localhost',
                    help='ip to connect to server')
PARSER.add_argument('--port',  type=str, default='6003',
                    help='port of the server')
PARSER.add_argument('--spectator',  action="store_true",
                    help='without interaction')
PARSER.add_argument('--session',  type=str, default=None,
                    help='session ID to join, if empty it will join the first session')
PARSER.add_argument('--carColor',  type=str, default=None,
                    help='color of the racecar')
PARSER.add_argument('--carName',  type=str, default=None,
                    help='displayed name of the racecar')
PARSER.add_argument('--model',  type=str, default=None,
                    help='chosen trained model')
PARSER.add_argument('--gameTrack',  type=int, default=None,
                    help='track number: 0,1,2')
PARSER.add_argument("--actionChoice", type=str, default="argmax",
                    help="Specifies how the model chooses its next action. Possible values: \"argmax\", \"possibilities\".")

# Argparse Decorator
def owlParser(func):
    """
    Decorator for enabling OwlRacer specific arguments to be parsed into the Environment
    """

    def wrapper():
        args = PARSER.parse_args()
        result = func(args)
        return result

    return wrapper
