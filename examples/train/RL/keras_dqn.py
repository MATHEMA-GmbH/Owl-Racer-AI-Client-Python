import os
from tensorflow import keras
import numpy as np
from owlracer import env
from owlracer.services import Command

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, GreedyQPolicy, MaxBoltzmannQPolicy, BoltzmannGumbelQPolicy
from rl.memory import SequentialMemory

ENV_NAME = 'RL-Training-OwlRacer'
GAME_TRACK = 1


def get_model(window_len: int = None):
    if not window_len:
        window_len = window_length
    print(env.observation_space.shape)

    model = Sequential()
    model.add(Flatten(input_shape=(window_len,) + env.observation_space.shape))
    model.add(Activation('relu'))
    model.add(Dense(28))
    model.add(Activation('relu'))
    model.add(Dense(14))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('softmax'))
    print(model.summary())

    return model


def get_dqn_agent():
    model = get_model()
    memory = SequentialMemory(limit=50000, window_length=window_length)
    policy = GreedyQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy,
                   test_policy=policy,
                   memory=memory,
                   nb_steps_warmup=62 * 2, gamma=0.95,
                   target_model_update=0.3,
                   train_interval=1, delta_clip=1.0)
    return dqn


def train_dqn():
    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    dqn = get_dqn_agent()
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    #checkpointer = ModelCheckpoint(filepath="/weights.hdf5", verbose=1, save_best_only=True)
    dqn.fit(env, nb_steps=50000, visualize=True, verbose=2, nb_max_episode_steps=1000,
            start_step_policy=1)

    # After training is done, we save the final weights.
    file_path = f'model_output/dqn_{ENV_NAME}_{GAME_TRACK}'
    os.makedirs("model", exist_ok=True)

    #dqn.model.save(f"{file_path}-a", save_format='h5', overwrite=True)
    #dqn.model.save(f"{file_path}-b", save_format='tf', overwrite=True)
    #dqn.save_weights(f"{file_path}.hdf5", overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=6000)
    play_dqn(dqn=dqn)


def play_dqn(dqn: DQNAgent):

    action = Command.idle

    print("Playing")
    for i in range(5000):
        observation, reward, terminated, info = env.step(action)
        if terminated:
            env.reset()
        action = dqn.forward(observation)
        print(action)


if __name__ == '__main__':
    # Get the environment and extract the number of actions.
    env = env.CarEnv(gameTrack=GAME_TRACK, carName="RL")
    env.action_space.seed(42)
    nb_actions = env.action_space.n
    print(f"nb_actions: {nb_actions}")

    # TODO: Reward anpassen/verändern, Policys, Hyperparameter, Modelarchitectur
    # Next, we build a very simple model.
    window_length = 1

    train_dqn()

    env.close()
