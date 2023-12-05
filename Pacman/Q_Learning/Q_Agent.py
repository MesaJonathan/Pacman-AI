import torch
import random
import numpy as np
from collections import deque #double ended queue
from Pacman import PacManAI, Direction, Point
from model import Linear_QNet, QTrainer
from graphs import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001    #learning rate

class Q_Agent:
    def __init__(self):
        self.n_games = 0    # number of games
        self.epsilon = 0    # controls randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, 256, 3) # 11 because..., hidden u can play w, 3 because either straight, left, or right
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    # gets current state of model, takes in the actual game itself
    def get_state(self, game):
        pass

    # idk what this does
    def remember(self, state, action, reward, next_state, done):
        pass

    #idrk what this one does either
    def train_long_memory(self):
        pass

    #idk what this really does
    def train_short_memory(self, state, action, reward, next_state, done):
        pass
    
    # Asks the model what the move is given this current state
    def get_action(self, state):
        final_move = [0, 0, 0, 0] #[up, down, left, right]

        return final_move


def train():
    plot_Q_scores = []
    plot_mean_Q_scores = []
    total_Q_score = 0
    Q_record = 0
    q_agent = Q_Agent()
    game = PacManAI()

    while True:
        # get old state
        state_old = q_agent.get_state(game)

        # get the move from the state
        final_move = q_agent.get_action(state_old)

        # do the move and get the new state
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