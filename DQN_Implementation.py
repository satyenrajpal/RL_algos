#!/usr/bin/env python
import numpy as np, gym, sys, copy, argparse
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Sequential,Model
import random
import tensorflow as tf
from collections import deque
from pathlib import Path
import keras
from keras import backend as K_back
from gym.wrappers import Monitor
import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pdb
import math
from PIL import Image

EPISODES=5000 #NUMBER OF EPISODES FOR LINEAR (with exp replay), DQN AND DUELING

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

def plot_eval(testX, testY):
		plt.title("Evaluation")
		plt.xlabel("Training Episode")
		plt.ylabel("Average Test Reward for 20 episodes")
		plt.plot(testX, testY)
		plt.show()

class QNetwork_keras():

	def __init__(self, env, model_type=None):
		self.learning_rate=0.0001														#######Hyperparameters
		self.obs_space=env.observation_space.shape[0]
		self.ac_space=env.action_space.n		
		
		if(model_type=='DQN'):
			print("Building DQN model")
			self.model=self.build_model_DQN()
		elif(model_type=='linear' or model_type=='Linear'):
			print("Building linear model")
			self.model=self.build_model_linear()
		elif(model_type=='Dueling' or model_type=='dueling'):
			print("Dueling  Model")
			self.model=self.build_model_dueling()
		else:
			print("Incorrect Model")
			assert 0==1
		
	def save_model_weights(self, name):
		# Helper function to save your model / weights. 
		self.model.save(name)

	def load_model(self, model_file):
		# Helper function to load an existing model.
		self.model=keras.models.load_model(model_file, custom_objects={"K_back": K_back})

	def load_model_weights(self,weight_file):
		# Helper funciton to load model weights. 
		self.model.load_weights(weight_file)

	def build_model_DQN(self):
		#Builds a DQN
		model=Sequential()
		model.add(Dense(units=24,input_dim=self.obs_space,activation='relu',
						kernel_initializer='he_uniform'))
		model.add(Dense(units=24,activation='relu',kernel_initializer='he_uniform'))
		model.add(Dense(units=self.ac_space,activation='linear',kernel_initializer='he_uniform'))
		model.summary()
		model.compile(loss='mean_squared_error',optimizer=Adam(lr=self.learning_rate))
		return model

	def build_model_dueling(self):
		inp=Input(shape=(self.obs_space,))
		x=Dense(units=32,activation='relu',kernel_initializer='he_uniform',name='hidden_layer_1')(inp)
		x=Dense(units=32,activation='relu',kernel_initializer='he_uniform',name='hidden_layer_2')(x)
		value_=Dense(units=1,activation='linear',kernel_initializer='he_uniform',name='Value_func')(x)
		ac_activation=Dense(units=self.ac_space,activation='linear',kernel_initializer='he_uniform',name='action')(x)
		#Compute average of advantage function
		avg_ac_activation=Lambda(lambda x: K_back.mean(x,axis=1,keepdims=True))(ac_activation)
		#Concatenate value function to add it to the advantage function
		concat_value=Concatenate(axis=-1,name='concat_0')([value_,value_])
		concat_avg_ac=Concatenate(axis=-1,name='concat_ac_{}'.format(0))([avg_ac_activation,avg_ac_activation])
		
		for i in range(1,self.ac_space-1):
			concat_value=Concatenate(axis=-1,name='concat_{}'.format(i))([concat_value,value_])
			concat_avg_ac=Concatenate(axis=-1,name='concat_ac_{}'.format(i))([concat_avg_ac,avg_ac_activation])
		
		#Subtract concatenated average advantage tensor with original advantage function
		ac_activation=Subtract()([ac_activation,concat_avg_ac])
		#Add the two (Value Function and modified advantage function)
		merged_layers=Add(name='final_layer')([concat_value,ac_activation])
		final_model=Model(inputs=inp,outputs=merged_layers)
		final_model.summary()
		final_model.compile(loss='mean_squared_error',optimizer=Adam(lr=self.learning_rate))
		return final_model

	
	def build_model_linear(self):
		#Builds a linear model
		model=Sequential()
		model.add(Dense(units=self.ac_space,input_dim=self.obs_space,kernel_initializer="he_uniform",
						activation="linear",use_bias=True))		
		model.summary()
		model.compile(loss='mean_squared_error',optimizer=Adam(lr=self.learning_rate))
		return model
	

class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=30000):

		self.burn_in=burn_in
		self.memory=memory_size
		self.mem_queue=deque(maxlen=self.memory)

	def sample_batch(self, batch_size=32):
		return random.sample(self.mem_queue,batch_size)

	def append(self, transition): 	
		if(len(self.mem_queue)<self.memory):
			self.mem_queue.append(transition)
		else:
			self.mem_queue.popleft()
			self.mem_queue.append(transition)

class DQN_Agent_ke():

	def __init__(self, env, render=False,model_type=None,save_folder=None):

		self.net=QNetwork_keras(env,model_type=model_type)
		self.obs_space=env.observation_space.shape[0]
		self.ac_space=env.action_space.n
		self.render=render
		######################Hyperparameters###########################
		self.env=env
		self.epsilon=0.7
		self.epsilon_min=0.05
		self.epsilon_decay=0.999
		self.gamma=0.99
		self.max_itr=1000000
		self.batch_size=32
		self.max_reward=160 #Used for saving a model with a reward above a certain threshold
		self.memory_queue=Replay_Memory(memory_size=50000, burn_in=30000)
		###############################################################
		self.avg_rew_buffer=10
		self.avg_rew_queue=deque(maxlen=self.avg_rew_buffer)
		self.model_save=50
		self.test_model_interval=50
		self.save_folder=save_folder
		
	def epsilon_greedy_policy(self, q_values,epsi):
		# Creating epsilon greedy probabilities to sample from.             
		if random.uniform(0,1)<=epsi:
			return random.randint(0,self.ac_space-1) #Q-Values shape is batch_size x ac
		else:
			return np.argmax(q_values[0])

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time. 
		return np.argmax(q_values[0]) 

	def train(self,exp_replay):
		testX,testY=[],[]
		batch_size,max_,avg_rew_test,itr=self.batch_size,0,0,0

		print("Using Experience Replay")
		#Burn In 
		self.burn_in_memory()

		if(self.save_folder!=None):
			self.env=Monitor(self.env, self.save_folder,video_callable=lambda episode_id:episode_id%500==0,force=True)		

		for epi in range(EPISODES):
			state=np.reshape(self.env.reset(),[1,self.obs_space])#Reset the state
			total_rew=0
			
			while True:
				itr+=1
				if(self.render):
					self.env.render()
				#get action by e-greedy
				ac=self.epsilon_greedy_policy(self.net.model.predict(state),self.epsilon)	
				#Find out next state and rew for current action
				n_s,rew,is_t, _ = self.env.step(ac) 
				##########################################################################
				#rew=-1 if is_t else rew
				###########################################################################
				#Append to queue
				n_s=np.reshape(n_s,[1,self.obs_space])
				self.memory_queue.append([state,ac,rew,is_t,n_s])
				#Get samples of size batch_size
				batch=self.memory_queue.sample_batch(batch_size=batch_size)

				#Create array of states and next states
				batch_states=np.zeros((len(batch),self.obs_space))
				batch_next_states=np.zeros((len(batch),self.obs_space))
				actions,rewards,terminals=[],[],[]

				for i in range(0,len(batch)):
					b_state, b_ac, b_rew, b_is_t, b_ns=batch[i] #Returns already reshaped b_state and b_ns
					batch_states[i]=b_state
					batch_next_states[i]=b_ns
					actions.append(b_ac)
					rewards.append(b_rew)
					terminals.append(b_is_t)

				#Get Predictions
				batch_q_values=self.net.model.predict(batch_states)
				batch_next_q_values=self.net.model.predict(batch_next_states)
				
				for i in range(0,len(batch)):
					if terminals[i]: #Corresponds to is_terminal in sampled batch
						batch_q_values[i][actions[i]]=rewards[i]

					else: 
					#If not
						batch_q_values[i][actions[i]]=rewards[i]+self.gamma*(np.amax(batch_next_q_values[i]))  
				#Perform one step of SGD
				self.net.model.fit(batch_states,batch_q_values,batch_size=batch_size,epochs=1,verbose=0)
				self.epsilon*=self.epsilon_decay
				self.epsilon=max(self.epsilon,self.epsilon_min)
				total_rew+=rew
				state=n_s
				
				if is_t:
					break
			
			#test model at intervals
			if((epi+1)%self.test_model_interval==0):
				testX.append(epi)
				avg_rew_test=self.test()
				testY.append(avg_rew_test)

			#Remove and add rewards to calculate avg reward
			if(len(self.avg_rew_queue)>self.avg_rew_buffer):
				self.avg.rew_queue.popleft()
			self.avg_rew_queue.append(total_rew)
			avg_rew=sum(self.avg_rew_queue)/len(self.avg_rew_queue)
			
			######################SAVING SECTION###############################
			#Save at intervals
			#if((epi+1)%self.model_save==0):
			#	self.net.model.save('CartPole_linearwExpReplay_{}.h5'.format(epi))
			if max_<avg_rew_test and avg_rew_test>self.max_reward:
				#self.net.model.save('CartPole_linear_comp_8.h5')
				max_=avg_rew_test
			######################################################################
			print(epi,itr,avg_rew,total_rew)

		plot_eval(testX,testY) #Plotting after episodes are done
			

	def test(self, model_file=None):
		test_episodes=20
		rewards=[]
		if(model_file!=None):
			self.net.load_model(model_file)
		for e in range(test_episodes):
			state = np.reshape(self.env.reset(),[1,self.obs_space])
			time_steps = 0
			total_reward_per_episode = 0
			while True:
				if(self.render):
					self.env.render()
				action = self.epsilon_greedy_policy(self.net.model.predict(state),0.05)
				next_state, reward, is_t, _ = self.env.step(action)
				next_state=np.reshape(next_state,[1,self.obs_space])
				state = next_state
				total_reward_per_episode+=reward
				time_steps+=1
				if is_t:
					break
			rewards.append(total_reward_per_episode)
			print("Total Reward for Episode {} is {}".format(e,total_reward_per_episode))
		
		avg_rewards_=np.mean(np.array(rewards))
		std_dev=np.std(rewards)
		print("AvgRew={},Std={}".format(avg_rewards_,std_dev))
		return avg_rewards_

	def burn_in_memory(self):
		# Initialize replay memory with a burn_in number of episodes / transitions. 
		memory_size=0
		state=np.reshape(self.env.reset(),[1,self.obs_space])
		
		while(memory_size<self.memory_queue.burn_in):
			ac=random.randint(0,self.ac_space-1)
			n_s,rew,is_t,_=self.env.step(ac)
			n_s=np.reshape(n_s,[1,self.obs_space])
			
			transition=[state,ac,rew,is_t,n_s]
			self.memory_queue.append(transition)
			state=n_s
			if is_t:
				state=np.reshape(self.env.reset(),[1,self.obs_space])
			memory_size+=1

		print("Burned Memory Queue")


#######################################PYTORCH IMPLEMENTATION#########################################
class QNetwork_py(nn.Module):

	# This class defines the network architecture.  
	def __init__(self, env_name):
		# Architecture of model is defined here
		nn.Module.__init__(self)
		self.env = gym.make(env_name)

		self.env_name = env_name
		self.state_size = self.env.observation_space.shape[0]
		self.action_size = self.env.action_space.n
		self.l1 = nn.Linear(self.state_size, self.action_size)

	def forward(self, x):
		x = self.l1(x)
		return x

	def save_model(self, checkpoint):
		torch.save(checkpoint, 'SavedModels/'+self.env_name+'-Linear.pt')

	def load_model(self, model_file):
		return torch.load(model_file)['state_dict']

class DQN_Agent_py():
	#Agent is trained in this class 
	def __init__(self, env_name, render=False,save_folder=None):

		self.env = gym.make(env_name)
		self.env_name = env_name
		self.model = QNetwork_py(env_name)
		if use_cuda:
			self.model.cuda()
		self.iterCount=0
		self.replay = Replay_Memory()
		if save_folder!=None:
			self.env = Monitor(self.env, 'Video/'+self.env_name+'-Linear-noReplay', force=True)

		####################################Hyper Parameters############################################
		if env_name == 'CartPole-v0':
			self.env_ind = 'C'
			self.nEpisodes = 1000
			self.EpisodeLength = 200
			self.discount = 0.99
			self.testDiscount = 1
			self.lr = 0.0001
			self.epsilon = 0.9
			self.epsilon_start = 0.9
			self.epsilon_decay = 2000
			self.epsilon_end = 0.05
			self.batch_size = 32
			self.testEpisodes = 20
			self.testInterval = 50
			self.epsilon_threshold = self.epsilon_start
			self.finalRun=100
		elif env_name == 'MountainCar-v0':
			self.env_ind = 'M'
			self.nEpisodes = 10000
			self.EpisodeLength = 200
			self.discount = 1
			self.testDiscount = 1
			self.lr = 0.001
			self.epsilon = 0.75
			self.epsilon_start = 1
			self.epsilon_decay = 10000
			self.epsilon_end = 0.05
			self.batch_size = 32 
			self.testEpisodes = 25	
			self.testInterval = 100	
			self.epsilon_threshold = self.epsilon_start
			self.finalRun=100
		################################################################################################

	def greedy_policy(self, state):
		return self.model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1,1)

	def epsilon_greedy_policy(self, state):
		p = random.random()

		self.epsilon_threshold*=0.9999
		if p > self.epsilon_threshold:
			return self.model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1,1)
		else:
			return FloatTensor([[random.randrange(2)]]) 

	def train(self):
		
		self.optimizer = optim.Adam(self.model.parameters(), self.lr)
		self.criterion = nn.MSELoss()
		testX = []
		testY = []
		best_reward = -1*math.inf
		reachedOnce = 0

		for e in range(self.nEpisodes):

			state = self.env.reset()
			total_reward ,time_steps= 0,0
			done = False

			while not done:
				self.iterCount+=1
				# if reachedOnce:
				# self.env.render()
				action = self.epsilon_greedy_policy(FloatTensor([state]))
				action_idx = action[0,0]
				next_state, reward, done, _ = self.env.step(int(action[0, 0]))
				if self.env_ind == 'C':
					if done and time_steps < 199:
						reward=-1
				curr_q_all = self.model(Variable(FloatTensor([state])))
				curr_q = curr_q_all[0,int(action_idx)]
				max_next_q = self.model(Variable(FloatTensor([next_state]))).max(1)[0]
				# print("Next q: {}".format(max_next_q))
				expected_q = reward + self.discount*max_next_q
				expected_q = Variable(expected_q.data, requires_grad=False)
				loss = self.criterion(curr_q, expected_q)
				# print("Loss: {}".format(loss))
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				total_reward+=reward
				state = next_state

			if self.env_ind=='C':
				print("{3}Episode:TR:epsilon {0}:{1}:{2}".format((e+1), total_reward, self.epsilon_threshold, '\033[91m' if self.env_ind=='C' and total_reward >= 195 else '\033[99m'))
			else:
				print("{3}Episode:TR:epsilon {0}:{1}:{2}".format((e+1), total_reward, self.epsilon_threshold, '\033[91m' if self.env_ind=='M' and total_reward > -200 else '\033[99m'))
			print("\x1b[0m")
			if (e+1)%self.testInterval==0:
				training_episode,avg_reward =  self.test(e+1)
				testX.append(training_episode)
				testY.append(avg_reward)

		plot_eval(testX, testY)
			# print("Episode:TR: epsilon {}:{}:{}".format((e+1), total_reward, self.epsilon_threshold))
	def test(self, training_episode, model_file=None):
		# Evaluate the performance of the agent over testepisodes, by calculating cummulative rewards for the episodes.
		print("\033[1;32;40m ##### Evaluation starts #####")
		total_reward = 0
		for e in range(self.testEpisodes):
			state = self.env.reset()
			done = False
			time_steps = 0
			total_reward_per_episode = 0
			while not done:
				# self.env.render()
				action = self.greedy_policy(FloatTensor([state]))
				action_idx = action[0,0]
				next_state, reward, done, _ = self.env.step(int(action[0,0]))
				state = next_state
				total_reward_per_episode+=reward
				time_steps+=1
			total_reward += total_reward_per_episode
			print("Total Reward for Episode {} is {}".format(e,total_reward_per_episode))
		avg_reward = total_reward/self.testEpisodes
		print("\033[1;32;40m ##### Evaluation ends #####")
		print("\x1b[0m")
		return training_episode,avg_reward
	
	def finalRunOnTrained(self,model_file):
		print("\033[1;32;40m ##### Final Run starts #####")
		total_reward = 0
		# savedModel = QNetwork_py(self.env_name)
		# pdb.set_trace()
		checkpoint = torch.load(model_file)
		self.model.load_state_dict(checkpoint['state_dict'])
		reward_list=[]
		for e in range(self.finalRun):
			state = self.env.reset()
			done = False
			time_steps = 0
			total_reward_per_episode = 0
			while not done:
				# self.env.render()
				# action = savedModel(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1,1)
				action = self.greedy_policy(FloatTensor([state]))
				action_idx = action[0,0]
				next_state, reward, done, _ = self.env.step(int(action[0,0]))
				state = next_state
				total_reward_per_episode+=reward
				time_steps+=1
			reward_list.append(total_reward_per_episode)
			total_reward += total_reward_per_episode
			print("Total Reward for Episode {} is {}".format(e,total_reward_per_episode))
		rewardArr = np.array(reward_list)
		print("Mean:{0}, Std:{1}".format(np.mean(rewardArr), np.std(rewardArr)))
		avg_reward = total_reward/self.testEpisodes
		print("\033[1;32;40m ##### Final Run ends #####")
		print("\x1b[0m")


#################################ATARI IMPLEMENTATION######################################
class QNetwork_atari(nn.Module):

	def __init__(self, env_name):
		nn.Module.__init__(self)
		self.env = gym.make(env_name)
		self.env_name=env_name
		self.state_size = self.env.observation_space.shape[0]
		self.action_size = self.env.action_space.n
		if env_name=='SpaceInvaders-v0':
			self.conv1 = nn.Conv2d(4,16,8,4)
			self.conv2 = nn.Conv2d(16,32,4,2)
			self.fc1 = nn.Linear(2592,256)
			self.fc2 = nn.Linear(256, self.action_size)
		else:
			self.l1 = nn.Linear(self.state_size, 64)
			self.l2 = nn.Linear(64, 128)
			self.l3 = nn.Linear(128,256)
			self.l4 = nn.Linear(256, self.action_size)


	def forward(self, x):
		if self.env_name=='SpaceInvaders-v0':
			# pdb.set_trace()
			x = F.relu(self.conv1(x))
			x = F.relu(self.conv2(x)).view(-1,2592)
			x = self.fc1(x)
			x = self.fc2(x)
		else:
			x = F.relu(self.l1(x))
			x = F.relu(self.l2(x))
			x = F.relu(self.l3(x))
			x = self.l4(x)
		return x

	def save_model(self, checkpoint, n_updates):
		torch.save(checkpoint, 'SavedModels/'+self.env_name+'-DDQN'+str(n_updates)+'.pt')

	def load_model(self, model_file):
		return torch.load(model_file)['state_dict']

class DQN_Agent_atari():
	
	def __init__(self, env_name, render=False):

		self.env = gym.make(env_name)
		self.action_size = self.env.action_space.n
		self.model = QNetwork_atari(env_name)
		self.target_network = QNetwork_atari(env_name)
		if use_cuda:
			self.model.cuda()
			self.target_network.cuda()
		self.iterCount=0
		self.replay = Replay_Memory()
		self.preprocess = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Grayscale(),
			transforms.Resize((110,84)),
			transforms.CenterCrop((84,84)),
			transforms.ToTensor(),
			])
		########################HYPERPARAMTERS#################################
		self.env_ind = 'S'
		self.discount = 0.8
		self.lr = 0.0001
		self.epsilon = 0.9
		self.epsilon_start = 0.9
		self.epsilon_stop = 100000
		self.epsilon_decay = 100000
		self.epsilon_end = 0.1
		self.batch_size = 8
		self.state_size = (84,84,4)		
		self.checkpoint_freq = 50000
		self.eval_interval= 100
		self.testEpisodes=20
		self.epsilon_threshold=1
		######################################################################

	def concatStates(self,state,action):
		state = self.preprocess(state).view(-1,1,84,84)
		total_reward = 0
		for i in range(3):
			next_state, reward, done, _ = self.env.step(action)
			next_state = self.preprocess(next_state).view(-1,1,84,84)
			state = torch.cat([state, next_state], 1)
			total_reward+=reward
		if use_cuda:
			return state.cuda()
		else:
			return state

	def greedy_policy(self, state):
		return self.model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1,1)

	def epsilon_greedy_policy(self, state, iterCount):
		p = random.random()
		if self.epsilon_threshold > self.epsilon_end:
			self.epsilon_threshold = self.epsilon_start - iterCount/self.epsilon_stop
			
		if p > self.epsilon_threshold:
			return self.model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1,1)
		else:
			return FloatTensor([[random.randrange(self.action_size)]])  

	def train(self):
		
		self.optimizer = optim.RMSprop(self.model.parameters(), self.lr)
		self.criterion = nn.MSELoss()
		max_updates = 1000000
		n_updates = 0
		eval_now=0
		n_episodes=0
		total_reward = 0
		curr_episode_updates = 0
		testX,testY,trainX,trainY=[],[],[],[]
		while n_updates<max_updates:

			state = self.env.reset()
			# action = FloatTensor([[random.randrange(self.action_size)]])
			reward_per_episode = 0
			done = False
			time_steps = 0
			best_reward = 0
			
			while not done:
				# self.env.render()
				# pdb.set_trace()
				if n_updates > 0:
					skipStates = nextSkipStates	
				else:
					action = FloatTensor([[random.randrange(self.action_size)]])
					action_idx = int(action[0,0])
					skipStates = self.concatStates(state, action_idx)
				action = self.epsilon_greedy_policy(skipStates, n_updates)
				# action = self.epsilon_greedy_policy(FloatTensor(skipStates), n_updates)
				action_idx = int(action[0,0])
				next_state, reward, done, _ = self.env.step(action_idx)
				# next_state, reward, done, _ = self.env.step(action)
				# pdb.set_trace()
				if use_cuda:
					nextSkipStates = torch.cat([skipStates[:,1:,:,:].cuda(),self.preprocess(next_state).view(-1,1,84,84).cuda()],1)
				else:
					nextSkipStates = torch.cat([skipStates[:,1:,:,:],self.preprocess(next_state).view(-1,1,84,84)],1)
				self.replay.append([skipStates.type(ByteTensor),action.type(ByteTensor), nextSkipStates.type(ByteTensor), ByteTensor([int(reward)])])
				# nextSkipStates, reward4states = self.concatStates(next_state, action_idx)

				if len(self.replay.mem_queue)>=self.replay.burn_in:
					transitions = self.replay.sample_batch(self.batch_size)
					batch_state = FloatTensor(self.batch_size,skipStates.shape[1],skipStates.shape[2],skipStates.shape[3])
					batch_action = FloatTensor(self.batch_size,1)
					batch_next_state = FloatTensor(self.batch_size,skipStates.shape[1],skipStates.shape[2],skipStates.shape[3])
					batch_reward = FloatTensor(self.batch_size,1)
					for i in range(len(transitions)):
						batch_state[i] = transitions[i][0].type(FloatTensor)
						batch_action[i] = transitions[i][1].type(FloatTensor)
						batch_next_state[i] = transitions[i][2].type(FloatTensor)
						batch_reward[i] = transitions[i][3].type(FloatTensor)
					batch_state = Variable(batch_state)
					batch_action = Variable(batch_action)
					batch_next_state = Variable(batch_next_state)
					batch_reward = Variable(batch_reward)

					## Change for Double DQN - start
					Q1_all_actions = self.model(batch_state)
					Q1 = Q1_all_actions.gather(1,batch_action.type(LongTensor))
					curr_q = Q1
					Q1_next_max, best_action = self.model(batch_next_state).max(1)
					Q2 = self.target_network(batch_next_state).gather(1,best_action.type(LongTensor).view(-1,1))
					expected_q = batch_reward + self.discount*Q2.view(self.batch_size,1)
					expected_q = Variable(expected_q.data, requires_grad=False)
					## Change for Double Q - end

					loss = self.criterion(curr_q, expected_q)
					self.optimizer.zero_grad()
					loss.backward()
					self.optimizer.step()
				n_updates+=1
				curr_episode_updates+=1
				reward_per_episode+=reward
				if n_updates%self.checkpoint_freq==0:
					checkpoint = {
					'episode': n_episodes,
					'state_dict': self.model.state_dict(),
					'optimizer': self.optimizer.state_dict()
					}
					self.model.save_model(checkpoint,n_updates)

			self.target_network.load_state_dict(self.model.state_dict())
			n_episodes+=1
			total_reward+=reward_per_episode
			print("{4}Episode:{0}, TR:{1}, epsilon:{2}, Updates:{3}".format((n_episodes), reward_per_episode, self.epsilon_threshold, curr_episode_updates,'\033[91m' if self.env_ind=='S' and reward_per_episode >= 300 else '\033[99m'))
			print("\x1b[0m")
			curr_episode_updates=0
			
			if n_episodes%self.eval_interval==0:

				training_episode,avg_test_reward =  self.test(n_episodes)
				testX.append(training_episode)
				testY.append(avg_test_reward)
				trainX.append(n_episodes)
				avg_training_reward = total_reward/self.eval_interval
				trainY.append(avg_training_reward)
				total_reward=0
				print("\033[1;34;40m ##### After Evaluating #####")
				print("Average Training Reward for last 100 episodes:{}".format(avg_training_reward))
				print("Average Test Reward for last 100 episodes:{}".format(avg_test_reward))
				print("\x1b[0m")

				with open('trainX.pickle','wb') as f:
					pickle.dump(trainX,f)
				with open('trainY.pickle','wb') as f:
					pickle.dump(trainY,f)
				with open('testX.pickle','wb') as f:
					pickle.dump(testX,f)
				with open('testY.pickle','wb') as f:
					pickle.dump(testY,f)

	def test(self, training_episode, model_file=None):
		print("\033[1;32;40m ##### Evaluation starts #####")
		total_reward = 0
		for e in range(self.testEpisodes):

			state = self.env.reset()
			n_updates=0
			done = False
			time_steps = 0
			total_reward_per_episode = 0
			while not done:
				if n_updates > 0:
					skipStates = nextSkipStates	
				else:
					action = FloatTensor([[random.randrange(self.action_size)]])
					action_idx = int(action[0,0])
					skipStates = self.concatStates(state, action_idx)
				action = self.epsilon_greedy_policy(skipStates, n_updates)
				action_idx = int(action[0,0])
				next_state, reward, done, _ = self.env.step(action_idx)
				if use_cuda:
					nextSkipStates = torch.cat([skipStates[:,1:,:,:].cuda(),self.preprocess(next_state).view(-1,1,84,84).cuda()],1)
				else:
					nextSkipStates = torch.cat([skipStates[:,1:,:,:],self.preprocess(next_state).view(-1,1,84,84)],1)

				total_reward_per_episode+=reward
				time_steps+=1
			total_reward += total_reward_per_episode
			print("Total Reward for Episode {} is {}".format(e,total_reward_per_episode))
		avg_reward = total_reward/self.testEpisodes
		print("\033[1;32;40m ##### Evaluation ends #####")
		print("\x1b[0m")
		return training_episode,avg_reward

def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--type',dest='model_type',type=str)
	parser.add_argument('--save_folder',dest='save_folder',type=str,default=None)
	parser.add_argument('--model_file',dest='model_file',type=str,default=None)
	return parser.parse_args()


def main(args):

	args = parse_arguments()
	env = gym.make(args.env)
	
	# gpu_ops = tf.GPUOptions(allow_growth=True)
	# config = tf.ConfigProto(gpu_options=gpu_ops)

	if(args.model_type=='linear'):
		agent=DQN_Agent_py(args.env)
		if(args.train):
			print("Training Linear Model without Experience Replay")
			agent.train()
	elif(args.model_type=='linear-exp'):
		if(args.train):
			model=DQN_Agent_ke(env,args.render,model_type=args.model_type,save_folder=args.save_folder)
			model.train(exp_replay=True)
	elif('SpaceInvaders' in args.env):
		agent=DQN_Agent_atari(args.env)
		agent.train()
	else:	
		if(args.train):
			("Training {} Model with Experience Replay".format(args.model_type))
			model = DQN_Agent_ke(env,args.render,model_type=args.model_type,save_folder=args.save_folder)
			model.train(exp_replay=True)
	
	#Test only if model_file has been given as an input
	if(args.model_file!=None):			
		file_=Path(args.model_file)
		try:
			abs_path=file_.resolve()
		except FileNotFoundError:
			print('File Not found')
		else:
			model.test(model_file=abs_path)

if __name__ == '__main__':
	main(sys.argv)

