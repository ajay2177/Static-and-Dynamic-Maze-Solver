import logging
from enum import Enum, IntEnum
import numpy as np
import read_maze


class Cell(IntEnum):
    EMPTY = 1       # Agent can move to this cell
    WALLFIRE = 0        # Agent cannot move to this cell

class Action(IntEnum):
    MOVE_LEFT = 0    # Move to the cell on the left
    MOVE_RIGHT = 1   # Move to the cell on the right
    MOVE_UP = 2      # Move to the cell at the top
    MOVE_DOWN = 3    # Move to the cell at the bottom
    STAY = 4         # Stay in the cell

class Status(Enum):
    WIN = 0      #To indicate whether the agent won
    LOSE = 1     #To indicate whether the agent lost
    PLAYING = 2  #To indicate whether the agent exceeded the maximum number of steps in an episode and lost


class Dynamic_Maze:

    actions = [Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.MOVE_UP, Action.MOVE_DOWN, Action.STAY]  # all actions

    reward_exit = 10.0  # reward for reaching the exit cell
    penalty_move = -0.05  # penalty for a move which did not result in finding the exit cell
    penalty_visited = -0.25  # penalty for returning to a cell which was visited earlier
    penalty_impossible_move = -0.75  # penalty for trying to enter an occupied cell or moving out of the maze

    def __init__(self, maze_size,  start_cell=(0,0), exit_cell=None):
        # Initialising the maze environment
        self.maze_size = maze_size
        self.rows = maze_size[0]
        self.cols = maze_size[1]
        self.__minimum_reward = -0.5 * self.rows * self.cols  # a threshold for stopping the game
        print("Min Reward(Threshold):", self.__minimum_reward)

        nrows = maze_size[0]
        ncols = maze_size[1]
        self.__start_cell = (0, 0) if start_cell is None else start_cell
        self.__exit_cell = (ncols - 1, nrows - 1) if exit_cell is None else exit_cell
        self.reset(self.__start_cell)

    def reset(self, start_cell):
        print("Start Cell: ", start_cell)
        # To reset the maze to its initial state and place the agent at the starting cell

        self.__previous_cell = self.__current_cell = start_cell
        self.__total_reward = 0.0  # total reward collected in the game
        self.__visited = set()  # to store the visited cells

        return self.__observe()

    def step(self, action):
        # To move the agent to the new state
        reward = self.__execute(action)
        self.__total_reward += reward
        status = self.__status()
        state = self.__observe()
        logging.debug("action: {:10s} | reward: {: .2f} | status: {}".format(Action(action).name, reward, status))
        return state, reward, status

    def __execute(self, action):
        # To execute the action and compute the reward
        possible_actions = self.__possible_actions(self.__current_cell)

        if not possible_actions:
            reward = self.__minimum_reward - 1  # cannot move, therefore end the game
        elif action in possible_actions:
            col, row = self.__current_cell
            if action == Action.MOVE_LEFT:
                col -= 1
            elif action == Action.MOVE_UP:
                row -= 1
            if action == Action.MOVE_RIGHT:
                col += 1
            elif action == Action.MOVE_DOWN:
                row += 1

            # Copying the cell states
            self.__previous_cell = self.__current_cell
            self.__current_cell = (col, row)

            if self.__current_cell == self.__exit_cell:
                reward = Dynamic_Maze.reward_exit  # maximum reward for reaching the exit cell
            elif self.__current_cell in self.__visited:
                reward = Dynamic_Maze.penalty_visited  # penalty for moving to an earlier visited cell
            elif self.__current_cell != self.__exit_cell:
                reward = Dynamic_Maze.penalty_move  # penalty for moving to a non-exit cell
            else:
                reward = 0 # penalty for staying in the same cell
            self.__visited.add(self.__current_cell)
        else:
            reward = Dynamic_Maze.penalty_impossible_move  # penalty for entering a cell with wall or fire

        return reward

    def __possible_actions(self, cell=None):
        # Defines all the possible actions
        if cell is None:
            col, row = self.__current_cell
        else:
            col, row = cell

        possible_actions = Dynamic_Maze.actions.copy()  # Initialise with all the actions

        # Remove the impossible actions to avoid the cells with wall or fire
        around_info = read_maze.get_local_maze_information(row, col)
        if row == 0 or (row > 0 and around_info[0, 1, 0] == Cell.WALLFIRE) or (row > 0 and around_info[0, 1, 1] >= 1):
            possible_actions.remove(Action.MOVE_UP)
        if row == self.maze_size[0] - 1 or (row < self.maze_size[0] - 1 and around_info[2, 1, 0] == Cell.WALLFIRE) or (row < self.maze_size[0] - 1 and around_info[2, 1, 1] >= 1):
            possible_actions.remove(Action.MOVE_DOWN)
        if col == 0 or (col > 0 and around_info[1, 0, 0] == Cell.WALLFIRE) or (col > 0 and around_info[1, 0, 1] >= 1):
            possible_actions.remove(Action.MOVE_LEFT)
        if col == self.maze_size[1] - 1 or (col < self.maze_size[1] - 1 and around_info[1, 2, 0] == Cell.WALLFIRE) or (col < self.maze_size[1] - 1 and around_info[1, 2, 1] >= 1):
            possible_actions.remove(Action.MOVE_RIGHT)
        if (row > 0 and around_info[0, 1, 1] >= 1) and (row < self.maze_size[0] - 1 and around_info[2, 1, 1] >= 1) and (col > 0 and around_info[1, 0, 1] >= 1) and (col < self.maze_size[1] - 1 and around_info[1, 2, 1] >= 1):
            possible_actions.remove(Action.MOVE_UP)
            possible_actions.remove(Action.MOVE_DOWN)
            possible_actions.remove(Action.MOVE_LEFT)
            possible_actions.remove(Action.MOVE_RIGHT)

        return possible_actions

    def __status(self):
        # Returns the status of the game - whether the agent has won or lost.
        if self.__current_cell == self.__exit_cell: # Reached the exit cell, agent has won the game
            return Status.WIN

        if self.__total_reward < self.__minimum_reward:  # End the game if the loss exceeds the threshold
            return Status.LOSE

        return Status.PLAYING

    def __observe(self):
        # Returns the current location of the agent
        return np.array([[*self.__current_cell]])

    def play(self, model, start_cell):
        # To play a single game after training the model. It uses the updated Q table.
        self.reset(start_cell)
        state = self.__observe()

        while True:
            action = model.predict(state=state)
            print(Action(action))
            state, reward, status = self.step(action)
            if status in (Status.WIN, Status.LOSE):
                return status
