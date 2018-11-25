import gym
import numpy as np 
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

from torch.autograd import Variable
from collections import namedtuple
from torch.distributions import Categorical

class A2C(nn.Module):

	def __init__(self, num_states = 4, num_actions = 2):
		super(A2C, self).__init__()

		# The first hidden layer remains same for both the actor and critic NN
		self.layer_1 = nn.Sequential(nn.Linear(num_states, 128), nn.ReLU(),)

		# Actor outputs a policy function, that is, state-action mapping function
		self.action_head = nn.Linear(128, num_actions)
		# Critic evaluates how good an action is, that is, quality of action taken
		self.critic_head = nn.Linear(128,1)

		# We save the actions and rewards that are used to update the weights of the neural network
		self.saved_actions = []
		self.rewards = []

	def forward(self, x):
		y = self.layer_1(x)

		action_scores = F.softmax(self.action_head(y) , dim = -1)
		state_values = self.critic_head(y)

		return action_scores, state_values

def select_action(state):
	state= torch.from_numpy(state).float()
	action_scores, state_values = model(state)
	m = Categorical(action_scores)
	action = m.sample()
	model.saved_actions.append(SavedAction(m.log_prob(action), state_values))

	return action.item()

def back_propagate():
	R = 0
	saved_actions = model.saved_actions
	policy_losses = []
	value_losses = []
	rewards = []
	for r in model.rewards[::-1]:
		R = r + gamma*R 
		rewards.insert(0, R)
	rewards = torch.tensor(rewards)
	rewards = (rewards- rewards.mean())/(rewards.std() + eps)
	for (log_prob, value), r in zip(saved_actions, rewards):
		reward = r - value.item()
		policy_losses.append(-log_prob * reward)
		value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))

	optimiser.zero_grad()
	total_loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
	total_loss.backward()
	optimiser.step()

	del model.rewards[:]
	del model.saved_actions[:]

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

env = gym.make("CartPole-v0")
env.seed(14)
torch.manual_seed(14)

gamma = 0.99
lr = 3e-2
num_episodes = 2000


num_actions = env.action_space.n
num_states = 4
render = False
eps = np.finfo(np.float32).eps.item()

model = A2C(num_states, num_actions)
optimiser = optim.Adam(model.parameters(), lr = lr)

episode_list = []
reward_list = []
running_reward_list = []


def main():
	running_reward = 10
	for episode in range(500):
		state = env.reset()
		for t in range(500):  
			action = select_action(state)
			state, reward, done, info = env.step(action)
			if render:
				env.render()
			model.rewards.append(reward)
			if done:
				break

		running_reward = running_reward * 0.99 + t * 0.01
		back_propagate()
		if episode % 10 == 0:
			print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                episode, t, running_reward))
			print("-----------------------------------------")
			print("Episode: ", episode)
			print("Last Episode Reward: ", t)


			episode_list.append(episode)
			reward_list.append(t)
		if running_reward > env.spec.reward_threshold:
			print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
			break


if __name__ == '__main__':
    main()

print("\n-------------Testing--------------\n")

def test_model(model):
	score = 0
	done = False
	env = gym.make('CartPole-v0')
	state = env.reset()
	while not done:
		action = select_action(state)
		state, reward, done, info = env.step(action)
		score +=reward
		if done:
			break
	return score

total = 0
for i in range(100):
	score = test_model(model)
	print("Score = ", score)
	total += score

print("AVERAGE SCORE of the trained A2C Model (over 100 trials):", total/100)

plt.plot(episode_list, reward_list)
plt.title("Episode Reward Variation")
plt.show()
