import torch
import random
import numpy as np
from collections import deque #double ended queue
from Playable_Pacman import Game, clock, pygame
from Q_Model import Q_Net, Q_Trainer
from itertools import chain
from graphs import Q_plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001    #learning rate


def train():
 
    game = Game(1,0)

    while True:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.running = False
                pygame.quit()
                quit()
        game.update()
        

if __name__ == "__main__":
    train()