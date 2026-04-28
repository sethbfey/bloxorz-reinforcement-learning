# Solving Bloxorz Using Deep Reinforcement Learning

## The Game

Bloxorz (also known as Roll the Block) is a challenging puzzle game. The objective is to reach the final square with a two-story block by rolling it vertically and horizontally without having it fall off the edge of the map. See [here](https://www.youtube.com/watch?v=1LaoH4I4iNQ) for a visual demonstration of the mechanics described.

The original version of the game developed by Damien Clarke in 2007 is no longer available on the internet but there are several remakes. This project replicates all 33 levels from the [CoolMath Games Bloxorz](https://www.coolmathgames.com/0-bloxorz) remake.

## The Repo

Contributions:
* **DQN Agent**: In this work, I created a simple double DQN network to sucessfully solve all 33 levels of Bloxorz. While successful, the Bloxorz agent did not always achieve the optimal move-sets. 
* **Bloxorz Level Dynamics**: Under `src/envs/levels/`, I recreated all 33 levels to its entirety using `.txt` for the tile layout and `.json` files for handling the dynamics of each mechanism.
* **A Web Demo**: I enlisted Claude to create a web server for manual play (see `src/scripts/play.py` for a simpler, 2D manual play) and to watch the DQN agents (see `src/sripts/replay.py` for a simpler, 2D replay). 
* **Scalability**: If desired, this framework warrants a straighforward creation schema and manual testing of custom Bloxorz levels.

## Setup

```
uv sync
```

Python 3.13.5+. PyTorch, Gymnasium, FastAPI, Pygame.

## Web demo

```
uv run uvicorn src.web.server:app --port 8000
```

Play any level or watch a trained agent. Frontend is a 3D isometric renderer on HTML5 Canvas.

## Training

```bash
uv run python src/agents/dqn.py --level 1 --<additional_arguments>
```

Checkpoints save to `runs/`.

## 2D Manual play

```
uv run python src/scripts/play.py --level 1
```

Arrow keys or WASD. `R` to restart and `Q` to quit.
