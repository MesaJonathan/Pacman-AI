# Retro Games
AI using Proximal Policy Optimization to play the game pacman made in PyGame by DevinLeamy.
I will be using his implementation of the classic game PacMan with the AI to play the game being written by myself.

# Pacman
The 80s classic Pacman in all its beauty <br/> <br/>
<!-- <img src="Pacman/Media/menu.png" alt="Pacman Menu Screen" width="400"/> -->

**How to Play:**
<br/>
1. Option One:
    1. Download the Pacman folder
    2. Download Python3 [https://www.python.org/downloads/]
    3. Install pygame(2.0.0) [pip3 and homebrew are easy options]
    4. In terminal, navigate to the file Pacman.py
    5. In terminal type python3 Pacman.py and hit enter
<br/>

<br/>
How was AI implemented
0. Environment Tuning
    1. get rid of title screen
    2. get rid of lives so there is one life and on death entire game resets(pellets, pacman, ghosts, etc...)
    

1. Q-learning
    rewards:
    - normal pellet: +3
    - power pellet: +5
    - eating ghost: + 10
    - death: -50

2. PPO
    1. pass
<br/>

Note: To adjust screen size change the variable "square" on line #59 of Pacman/Pacman.py <br/>

**Gameplay (delay is just an artifact of the video quality; it runs smooth):**
<br/>
<img src="Pacman/Media/gameplay.gif" alt="Pacman Gameplay" width="600"/>
