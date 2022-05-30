import logging
import random
from datetime import datetime

import numpy as np
from s_game import Status
# from d_game import Status

class QTableModel():

    def __init__(self, game, start_cell):
        # Initialise the Q learning algorithm
        self.game = game
        self.start_cell = start_cell
        self.Q = dict()  # Initialising the Q-table

    def train(self, **kwargs):
        # To train the model
        discount = kwargs.get("discount", 0.90)  # To determine the importance on future rewards
        exploration_rate = kwargs.get("exploration_rate", 0.10)  # Threshold to choose between exploration and exploitation
        exploration_decay = kwargs.get("exploration_decay", 0.995)  # To reduce the exploration rate every episode
        learning_rate = kwargs.get("learning_rate", 0.10)  # To determine the speed of learning, i.e. how much to update the old values with new values
        episodes = max(kwargs.get("episodes", 1000), 1)  # Total number of training episodes
        max_steps = kwargs.get("max_steps", 30000)  # Total number of steps per training episode
        # variables for storing the info
        cumulative_reward = 0
        cumulative_reward_history = []
        start_time = datetime.now()
        total_steps = 0
        print("Running Q learning")
        for episode in range(episodes): # For every episode

            state = self.game.reset(self.start_cell)
            state = tuple(state.flatten())
            step_count = 0

            for i in range(max_steps): # For every step in an episode
                # choose action based on epsilon greedy strategy
                if np.random.random() < exploration_rate:
                    action = random.choice(self.game.actions) # Choose Random action
                else:
                    action = self.predict(state) # Choose Max return action

                next_state, reward, status = self.game.step(action)
                next_state = tuple(next_state.flatten())

                cumulative_reward += reward

                if (state, action) not in self.Q.keys():
                    self.Q[(state, action)] = 0.0

                max_next_Q = max([self.Q.get((next_state, a), 0.0) for a in self.game.actions])
                # Update Q table for Q(s,a)
                self.Q[(state, action)] += learning_rate * (reward + discount * max_next_Q - self.Q[(state, action)])
                # To check win or lose status
                if status in (Status.WIN, Status.LOSE):
                    break

                step_count += 1
                # Assign current state as next state
                state = next_state

            #Reward for all episodes
            cumulative_reward_history.append(cumulative_reward)
            print("episode: {:d}/{:d} | status: {:4s} | e: {:.5f}".format(episode, episodes, status.name, exploration_rate))
            exploration_rate *= exploration_decay
            print('Steps', step_count)
            total_steps += step_count
        print("episodes: {:d} | time spent: {}".format(episode+1, datetime.now() - start_time))

        return cumulative_reward_history, episode, datetime.now() - start_time

    def q(self, state):
        # Get the Q values of all the actions for a particular state
        if type(state) == np.ndarray:
            state = tuple(state.flatten())

        return np.array([self.Q.get((state, action), 0.0) for action in self.game.actions])

    def predict(self, state):
        # To choose the action with the highest value from the Q table
        q = self.q(state)
        logging.debug("q[] = {}".format(q))

        actions = np.nonzero(q == np.max(q))[0]  # Index of the action/actions with the max value
        return random.choice(actions)
