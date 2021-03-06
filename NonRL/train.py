import gym
import numpy as np 
import tflearn
from tflearn.layers.estimator import regression
from tflearn.layers.core import dropout, fully_connected, input_data
import matplotlib.pyplot as plt


lr=0.01
env=gym.make('CartPole-v0')
env.reset()

#Dfnining the hyperparameters of the model

no_of_steps=500 # No. of the steps before a game ends by default
min_score=45# Minimum score above which the train game is included in the final training data
train_games=10000 #Number of games in training
model_optimizer='adam' 
model_loss='categorical_crossentropy'
no_of_episodes=100 #Number of episodes in testing 



#Creating a training dataset
def training_population():
	scores=[]
	training_data=[]
	scores_more_min=[]  #Scores that are greater than the selected threshold to be included in training dataset

	for a in range(train_games):
		score=0
		game_mem=[]
		pre_obs=[]
		for b in range(no_of_steps):
			action=env.action_space.sample()
			observations, rewards, done, info = env.step(action)
			if len(pre_obs)>0:
				game_mem.append([pre_obs, action]) #Creating a set of all observations

			pre_obs=observations
			score+=rewards
			if done:
				break

		#Include only those observation for which score>min_score	
		if score>=min_score:
			scores_more_min.append(score)
			for i in game_mem:
				if i[1]==1:
					output = [0 ,1]
				else:
					output = [1, 0]

				training_data.append([i[0], output])


		env.reset()
		scores.append(score)
	training_data=np.array(training_data)
	np.save('saved.npy', training_data)
	print('Average score:' , np.mean(scores_more_min))
	print('Median:' , np.median(scores_more_min))
	print(len(scores_more_min))
	print(scores_more_min)
	return training_data

	




#training_population()

#training_data=training_population()


#Defining the Neural Network - Fully connected with 5 layers
def neural_net(size):
	network=input_data(shape=[None, size, 1], name='input_layer')
	network=fully_connected(network, 128, activation='relu')
	dropout(network,0.6)
	network=fully_connected(network, 256, activation='relu')
	dropout(network,0.6)
	network=fully_connected(network, 512, activation='relu')
	dropout(network,0.6)
	network=fully_connected(network, 256, activation='relu')
	dropout(network,0.6)
	network=fully_connected(network, 128, activation='relu')
	dropout(network,0.6)

	network=fully_connected(network, 2, activation='softmax')

	network=regression(network, optimizer=model_optimizer, learning_rate=lr, loss=model_loss, name='output')

	model=tflearn.DNN(network)
	return model

#Training the neural network on the training dataset obtained earlier
def training_model(training_data, model=False):
	x=np.array([arr[0] for arr in training_data]).reshape(-1, len(training_data[0][0]),1)
	#for i in range(len(training_data)):
	#	x=np.array([training_data[i][0]]).reshape(-1,len(training_data[0][0]),1)
	y=[arr[1] for arr in training_data]

	if not(model):
		model=neural_net(len(x[0]))
    
	model.fit({'input_layer':x}, {'output':y},n_epoch=5, show_metric=True)

	return model


training_data=training_population()
model=training_model(training_data)


model.save('Test.model')

#model.load('Test.model')


#Testing our model on 100 games
scores=[]
action_set=[]

episode=np.empty(no_of_episodes)
i=0
for game in range(no_of_episodes):
	i+=1
	episode[i-1]=i
	score=0
	game_mem=[]
	pre_obv=[]
	env.reset()
	for a in range(no_of_steps):
		env.render()
		if len(pre_obv)==0:
			action = env.action_space.sample()
		else:
			action=np.argmax(model.predict(np.asarray(pre_obv).reshape(-1,len(pre_obv),1))[0])

		action_set.append(action)
		observation, reward, done, info=env.step(action)
		pre_obv=observation
		game_mem.append([observation, action])
		score+=reward
		if done:
			break

	scores.append(score)




print("Average score:", np.mean(scores))
print('Action 1:', action_set.count(1))
print('Action 0:', action_set.count(0))
plt.plot(episode, scores)
plt.show()




















