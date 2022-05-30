import logging
from enum import Enum, IntEnum
import matplotlib.pyplot as plt
import numpy as np


class Cell(IntEnum):

    EMPTY = 1       # Agent can move to this cell
    WALL = 0        # Agent cannot move to this cell

class Action(IntEnum):
    MOVE_LEFT = 0    #Move to the cell on the left
    MOVE_RIGHT = 1   #Move to the cell on the right
    MOVE_UP = 2      #Move to the cell at the top
    MOVE_DOWN = 3    #Move to the cell at the bottom


class Render(Enum):
    #For rendering the moves of the agent
    NOTHING = 0
    TRAINING = 1
    MOVES = 2


class Status(Enum):
    WIN = 0      #To indicate whether the agent won
    LOSE = 1     #To indicate whether the agent lost
    PLAYING = 2  #To indicate whether the agent exceeded the maximum number of steps in an episode and lost


class Static_Maze:

    actions = [Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.MOVE_UP, Action.MOVE_DOWN]  # all actions

    reward_exit = 10.0  # reward for reaching the exit cell
    penalty_move = -0.05  # penalty for a move which did not result in finding the exit cell
    penalty_visited = -0.25  # penalty for returning to a cell which was visited earlier
    penalty_impossible_move = -0.75  # penalty for trying to enter an occupied cell or moving out of the maze

    def __init__(self, maze,  start_cell=(0,0), exit_cell=None):
        # Initialising the maze environment
        self.maze = maze
        self.__minimum_reward = -0.5 * self.maze.size  # a threshold for stopping the game
        print("Min Reward(Threshold):", self.__minimum_reward)

        nrows, ncols = self.maze.shape
        self.cells = [(col, row) for col in range(ncols) for row in range(nrows)]
        self.empty = [(col, row) for col in range(ncols) for row in range(nrows) if self.maze[row, col] == Cell.EMPTY]
        self.__start_cell = (0, 0) if start_cell is None else start_cell
        self.__exit_cell = (ncols - 1, nrows - 1) if exit_cell is None else exit_cell
        self.empty.remove(self.__exit_cell)

        # Impossible maze layout Check
        if self.__exit_cell not in self.cells:
            raise Exception("Error: exit cell at {} is not inside maze".format(self.__exit_cell))
        if self.maze[self.__exit_cell[::-1]] == Cell.WALL:
            raise Exception("Error: exit cell at {} is not free".format(self.__exit_cell))

        # Rendering Variables
        self.__render = Render.NOTHING  # what to render
        self.__ax1 = None  # axes for rendering the moves
        self.__ax2 = None  # axes for rendering the best action per cell
        self.reset(self.__start_cell)

    def reset(self, start_cell):
        print("Start Cell: ", start_cell)
        # To reset the maze to its initial state and place the agent at the starting cell

        self.__previous_cell = self.__current_cell = start_cell
        self.__total_reward = 0.0  # total reward collected in the game
        self.__visited = set()  # to store the visited cells

        if self.__render in (Render.TRAINING, Render.MOVES):
            # render the maze
            nrows, ncols = self.maze.shape
            self.__ax1.clear()
            self.__ax1.set_xticks(np.arange(0.5, nrows, step=1))
            self.__ax1.set_xticklabels([])
            self.__ax1.set_yticks(np.arange(0.5, ncols, step=1))
            self.__ax1.set_yticklabels([])
            self.__ax1.grid(True)
            self.__ax1.plot(*self.__current_cell, "rs", markersize=5)  # start cell indicator
            self.__ax1.plot(*self.__exit_cell, "gs", markersize=5)  # exit cell indicator
            self.__ax1.imshow(self.maze, cmap="binary")
            self.__ax1.get_figure().canvas.draw()
            self.__ax1.get_figure().canvas.flush_events()

        return self.__observe()

    def __draw(self):
        # To draw the trajectory of the agent
        self.__ax1.plot(*zip(*[self.__previous_cell, self.__current_cell]), "bo-")  # previous cells indicator
        self.__ax1.plot(*self.__current_cell, "ro")  # current cell indicator
        self.__ax1.get_figure().canvas.draw()
        self.__ax1.get_figure().canvas.flush_events()

    def render(self, content=Render.NOTHING):
        # To render the moves of the agent
        self.__render = content

        if self.__render == Render.NOTHING:
            if self.__ax1:
                self.__ax1.get_figure().close()
                self.__ax1 = None
            if self.__ax2:
                self.__ax2.get_figure().close()
                self.__ax2 = None
        if self.__render == Render.TRAINING:
            if self.__ax2 is None:
                fig, self.__ax2 = plt.subplots(1, 1, tight_layout=True)
                fig.canvas.set_window_title("Action")
                self.__ax2.set_axis_off()
        if self.__render in (Render.MOVES, Render.TRAINING):
            if self.__ax1 is None:
                fig, self.__ax1 = plt.subplots(1, 1, tight_layout=True)
                fig.canvas.set_window_title("Maze")

        plt.show(block=False)
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

            if self.__render != Render.NOTHING:
                self.__draw()

            if self.__current_cell == self.__exit_cell:
                reward = Static_Maze.reward_exit  # maximum reward when reaching the exit cell
            elif self.__current_cell in self.__visited:
                reward = Static_Maze.penalty_visited  # penalty when returning to a cell which was visited earlier
            else:
                reward = Static_Maze.penalty_move  # penalty for a move which did not result in finding the exit cell

            self.__visited.add(self.__current_cell)
        else:
            reward = Static_Maze.penalty_impossible_move  # penalty for trying to enter an occupied cell or move out of the maze

        return reward

    def __possible_actions(self, cell=None):
        # Defines all the possible actions
        if cell is None:
            col, row = self.__current_cell
        else:
            col, row = cell

        possible_actions = Static_Maze.actions.copy()  # Initialise with all the actions

        # Remove the impossible actions
        nrows, ncols = self.maze.shape
        if row == 0 or (row > 0 and self.maze[row - 1, col] == Cell.WALL):
            possible_actions.remove(Action.MOVE_UP)
        if row == nrows - 1 or (row < nrows - 1 and self.maze[row + 1, col] == Cell.WALL):
            possible_actions.remove(Action.MOVE_DOWN)

        if col == 0 or (col > 0 and self.maze[row, col - 1] == Cell.WALL):
            possible_actions.remove(Action.MOVE_LEFT)
        if col == ncols - 1 or (col < ncols - 1 and self.maze[row, col + 1] == Cell.WALL):
            possible_actions.remove(Action.MOVE_RIGHT)

        return possible_actions

    def __status(self):
        # Returns the status of the game - whether the agent has won or lost.
        if self.__current_cell == self.__exit_cell: # Reached the exit cell, agent has won the game
            return Status.WIN

        if self.__total_reward < self.__minimum_reward:  # End the game if there is too much loss
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
