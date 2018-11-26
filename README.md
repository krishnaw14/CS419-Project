# CS419-Project
Analysis of different  deep Reinforcement Learning methods

## Repository Structure: 

random_action.py : Agent without training

### Non-RL
- train.py : tflearn code for solving cartpole environnment without using Reinforcement learning

### Deep_Qlearn
- deepq.py : pytorch code for solving cartpole environnment without using Deep Q-learning

### PolicyGradient
- cartpole-pg-tf : Policy Gradient where policy function is a neural network (written using tensorflow)
- cartpole-pg.py : Policy gradient where policy function is evaluated using dot product between randomly generated numbers and state of the env. Deep Learning is not used. 

### Actor-Critic
- a2c-cartpole.py : Advantage Actor-critic algorithm for solving the Cartpole environment

