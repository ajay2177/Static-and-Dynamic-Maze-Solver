import matplotlib.pyplot as plt
import numpy as np
from qtable import QTableModel
from s_game import Static_Maze, Render
from d_game import Dynamic_Maze
import read_maze
global maze_cells

# Reading the maze altogether in the beginning so that the training is fast. However this cannot be done in the case of Dynamic Maze
maze_size = (201,201)
read_maze.load_maze()
maze_env = np.zeros(maze_size)
for i in range(0, 201):
    for j in range(0, 201):
        maze_env[i, j] = read_maze.get_local_maze_information(i, j)[1, 1, 0]

print("Maze_env", maze_env)
start = (1,1)
exit = (maze_size[0]-2,maze_size[1]-2)
game = Static_Maze(maze=maze_env, start_cell= start, exit_cell = exit)
# game = Dynamic_Maze(maze_size=maze_size, start_cell= start, exit_cell = exit)
model = QTableModel(game, start_cell=start)
h, w, _ = model.train(discount=0.90, exploration_rate=0.2, exploration_decay = 0.995, learning_rate=0.10, episodes=2000, max_steps=50000)
print("Going to Play")
game.render(Render.MOVES)
game.play(model, start_cell=(1, 1))

plt.show()



