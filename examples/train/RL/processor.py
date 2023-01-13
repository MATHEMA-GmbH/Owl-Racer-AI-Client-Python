from rl.core import Processor
import numpy


class CustomProcessor(Processor):
    '''
    acts as a coupling mechanism between the agent and the environment
    '''

    def process_step(self, observation, reward, done, info):
        """Processes an entire step by applying the processor to the observation, reward, and info arguments.

        # Arguments
            observation (object): An observation as obtained by the environment.
            reward (float): A reward as obtained by the environment.
            done (boolean): `True` if the environment is in a terminal state, `False` otherwise.
            info (dict): The debug info dictionary as obtained by the environment.

        # Returns
            The tupel (observation, reward, done, reward) with with all elements after being processed.
        """
        print(f"!!!! info: {info}")

        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info

    def process_info(self, info):
        """Processes the info as obtained from the environment for use in an agent and
        returns it.

        # Arguments
            info (dict): An info as obtained by the environment

        # Returns
            Info obtained by the environment processed
        """
        print(f"info: {info}")
        return info

    def process_state_batch(self, batch):
        '''
        Given a state batch, I want to remove the second dimension, because it's
        useless and prevents me from feeding the tensor into my CNN
        '''
        print(f"batch: {batch}")
        #return numpy.squeeze(batch, axis=1)
        return batch
