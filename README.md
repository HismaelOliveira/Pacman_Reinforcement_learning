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

## Improvements

Even though it achieved good results, we found it interesting to reflect on what we could have tested more and improved. Our first thought relates to the score improvement: to the best of our knowledge, if we *use the pills in the state modeling*, we could reach better results. Our second thought was related to the *test of new parameters combinations* for each world, including the number of training episodes.
