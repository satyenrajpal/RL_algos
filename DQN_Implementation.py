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
from networks import QNetwork

EPISODES=5000 #NUMBER OF EPISODES 

def plot_eval(testX, testY):
		plt.title("Evaluation")
		plt.xlabel("Training Episode")
		plt.ylabel("Average Test Reward for 20 episodes")
		plt.plot(testX, testY)
		plt.show()

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

class DQN_Agent():

	def __init__(self, env, render=False,model_type=None,save_folder=None):

		self.net=QNetwork(env,model_type=model_type)
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

	def train(self):
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
	

	if(args.train):
		("Training {} Model with Experience Replay".format(args.model_type))
		model = DQN_Agent(env,args.render,model_type=args.model_type,save_folder=args.save_folder)
		model.train()
	
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

