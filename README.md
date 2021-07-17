# PacMan Agent using Approximate Q-Learning Algorithm 

## Algorithm

The **states** are based on the following features:
- Number of ghosts 1 step closer (north, south, west, east);
- Ghosts 2 steps closer;
- Food available 1 step closer;
- The nearest food;

The **terminal state** is the end of the game (if Pacman wins or if a ghost captures him).

The **actions** are: go north, go east, go south, go west or stop.

The **reward function** is calculated by the current score minus the last score.

## Execution

For the reinforcement learning algorithm, download the RL Agent (RlAgents.py, attached) and run the following code (for Reinforcement Learning algorithm), using the pacman library of CS188: 

`!python pacman.py -p RLAgent -x 1000 -n 1010 -l smallClassic`

* -p: the agent (RLAgent is the Reinforcement Learning Agent)

* -x: training iterations

* -n: test iterations

* -l: layout (choose between smallClassic, mediumClassic and originalClassic)

## Improvements

Even though it achieved good results, we found it interesting to reflect on what we could have tested more and improved. Our first thought relates to the score improvement: to the best of our knowledge, if we *use the pills in the state modeling*, we could reach better results. Our second thought was related to the *test of new parameters combinations* for each world, including the number of training episodes.
