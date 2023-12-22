import torch
import random
import numpy as np
from collections import deque #double ended queue
from Pacman import PacManAI
from Q_Model import Q_Net, Q_Trainer
from itertools import chain
#from graphs import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001    #learning rate

class Q_Agent:
    def __init__(self):
        self.n_games = 0    # number of games
        self.epsilon = 0    # controls randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Q_Net(1027, 256, 4) 
        self.trainer = Q_Trainer(self.model, lr=LR, gamma=self.gamma)

    # gets current state of model, takes in the actual game itself
    def get_state(self, game):
        # ghost states, ghost positions (row, col), ghost dirs
        ghost1 = game.ghosts[0]
        ghost2 = game.ghosts[1]
        ghost3 = game.ghosts[2]
        ghost4 = game.ghosts[3]

        ghost_positions = [(ghost1.row, ghost1.col), 
                           (ghost2.row, ghost2.col), 
                           (ghost3.row, ghost3.col), 
                           (ghost4.row, ghost4.col)]
        
        ghost_directions = [ghost1.dir,
                            ghost2.dir, 
                            ghost3.dir, 
                            ghost4.dir]
        
        # ghost_states = [ghost1.attacked,
        #                 ghost2.attacked,
        #                 ghost3.attacked,
        #                 ghost4.attacked]

        # pac man position and dir
        pacman_pos = (game.pacman.row, game.pacman.col)

        pacman_dir = game.pacman.dir
        
        # get what direction pellets are relative to pacman
        gb = game.gb

        # food_up = 0 
        # for row in range(int(pacman_pos[0])):
        #     if (2 in gb[row]) or (6 in gb[row]):
        #         food_up = 1

        # food_down = 0
        # for row in range(int(pacman_pos[0]), len(gb)):
        #     if (2 in gb[row]) or (6 in gb[row]):
        #         food_down = 1

        # food_left = 0
        # for row in gb:
        #     for col in range(int(pacman_pos[1])):
        #         if (2 in gb[row][col]) or (6 in gb[row][col]):
        #             food_left = 1

        # food_right = 0
        # for row in gb:
        #     for col in range((pacman_pos[1]), len(gb[row])):
        #         if (2 in gb[row][col]) or (6 in gb[row][col]):
        #             food_right = 1


        # pellets = [food_up, food_left, food_right, food_down]

        flat_board = list(chain.from_iterable(gb))

        #print(len(flat_board))

        state = flat_board + [game.pacman.row, # pacman pos
                 game.pacman.col, 
                 game.pacman.dir, # pac man dir
                 ghost1.row,      # ghost positions
                 ghost1.col, 
                 ghost2.row, 
                 ghost2.col, 
                 ghost3.row, 
                 ghost3.col, 
                 ghost4.row, 
                 ghost4.col, 
                 ghost1.dir,      # ghost directions
                 ghost2.dir, 
                 ghost3.dir, 
                 ghost4.dir, 
                 ghost1.attacked, #ghost states
                 ghost2.attacked,
                 ghost3.attacked,
                 ghost4.attacked]

        print(len(state))
        return state

    # idk what this does
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # training memory once pacman dies
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    # training memory every step of the game
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    # Asks the model what the move is given this current state
    def get_action(self, state):
        final_move = [0, 0, 0, 0] #[up, down, left, right]
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_Q_scores = []
    plot_mean_Q_scores = []
    total_Q_score = 0
    Q_record = 0
    q_agent = Q_Agent()
    game = PacManAI(0,1)

    while True:
        # get old state
        state_old = q_agent.get_state(game)

        # get the move from the state
        final_move = q_agent.get_action(state_old)

        # do the move and get the new state
        game.move = final_move
        reward, done, score = game.update(final_move)
        state_new = q_agent.get_state(game)

        # train the short memory ???
        q_agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember?
        q_agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            #train memory and plot the result
            game.game_reset()
            q_agent.n_games += 1 
            q_agent.train_long_memory()

            # high score
            if score > Q_record:
                Q_record = score
                #q_agent.model.save()
            
            #do graphing stuff here

            # plot_scores.append(score)
            # total_score += score
            # mean_score = total_score / agent.n_games
            # plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)

if __name__ == "__main__":
    train()