import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym
from keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras.backend as K
from collections import deque
from gym.wrappers import Monitor
import os
import pickle
_EPSILON=1e-7

def one_hot_vc(ac_space,curr_ac):
	 a=np.zeros(ac_space)
	 a[curr_ac]=1
	 return a
def plot_this(num_episodes,mean_,std_,his_len):
	x=np.linspace(0,num_episodes,num_episodes/his_len+1)
	plt.errorbar(x, mean_, yerr=std_,linestyle=None,marker='^')
	plt.savefig("Plot")
	plt.show()

def Reinforce_loss(y_true,y_pred):
	''' REINFORCE LOSS'''
	#y_true=[0,0,G_t,0]
	#y_pred=model.predict(states)=[probabilites of actions]
	epsilon=tf.convert_to_tensor(_EPSILON,y_pred.dtype.base_dtype)
	
	probable_acs=tf.clip_by_value(y_pred,epsilon,1-epsilon)
	loss=tf.multiply(y_true,tf.log(probable_acs))#[0*P(0|S),0*P(1|S),G_t*P(2|S),0*P(3|S)]
	loss_vector=K.sum(loss,axis=1) #0+0+G_t*P(2|S)+0 = G_t*P(2|S)
	loss=tf.reduce_mean(loss_vector)#self explanatory
	return -loss

class Reinforce(object):
	# Implementation of the policy gradient method REINFORCE.

	def __init__(self, model, lr,env):
		self.model = model
		# self.model.summary()
		self.save_folder=os.path.join(os.getcwd(),'videos')
		self.model.compile(loss=Reinforce_loss,
			 optimizer=Adam(lr=lr))

	def train(self, env, gamma=1.0,render=False):
		# Trains the model on a single episode using REINFORCE.
		his_len=500
		rew_history=deque(maxlen=his_len)
		step_history=deque(maxlen=his_len)
		env=Monitor(env, self.save_folder,video_callable=lambda episode_id:episode_id%500==0,force=True)
		mean_total=[]
		std_total=[]
		num_episodes=45000
		for l in range(num_episodes+1):
			# render=True if l%100==0 else False
			states,actions_OH,rewards,_=self.generate_episode(env,render)
			
			# G_t,decay=0,0
			G_total=[0]
			# G_total.append(rewards[-1]) #Appending G_(T-1)
			for i in reversed(rewards):
				G_t=i+gamma*G_total[-1]
				G_total.append(G_t)
			# print(actions_OH)
			G_total=G_total[1:]
			G_total=np.flip(np.array(G_total),axis=0)
			G_total=np.expand_dims(G_total,axis=-1)
			G_total_actions=np.multiply(G_total,actions_OH)
			history=self.model.fit(states,G_total_actions*1e-2,verbose=0,epochs=1,batch_size=len(states))

			if(len(rew_history)>=his_len):
				rew_history.popleft()
			if(len(step_history)>=his_len):
				step_history.popleft()
			step_history.append(len(states))
			rew_history.append(G_t)
			if(l%500==0):
				mean_rew,mean_steps=0,0
				for i in range(len(rew_history)):
					mean_rew+=rew_history[i]
					mean_steps+=step_history[i]
				mean_steps/=len(step_history)
				mean_rew/=len(rew_history)
				mean_100,std_100=self.test(env)
				mean_total.append(mean_100)
				std_total.append(std_100)
				if(l%500==0):
					print("Mean Rew={},Mean_steps={},Loss={}".format(mean_rew,mean_steps,history.history['loss']))
					print("Mean TEST={},STD_DEV_TEST={}".format(mean_100,std_100))
		#Plotting
		mean_total=np.expand_dims(np.array(mean_total),axis=-1)
		std_total=np.expand_dims(np.array(std_total),axis=-1)
		pickle.dump(mean_total,open("Reinforce_mean.pkl","wb"))
		pickle.dump(std_total,open("Reinforce_std.pkl","wb"))
		
		plot_this(num_episodes,mean_total,std_total,his_len)
			
	def test(self,env,num_episodes=100,render=False):
		rew_episodes=[]
		for _ in range(num_episodes):
			_,_,r,_=self.generate_episode(env,render=render)
			rew_episodes.append(np.sum(np.array(r)))
		rew_episodes=np.array(rew_episodes)
		return np.mean(rew_episodes),np.std(rew_episodes)

	def generate_episode(self, env, render=False):
		# Generates an episode by executing the current policy in the given env.
		# Returns:
		# - a list of states, indexed by time step
		# - a list of actions, indexed by time step
		# - a list of rewards, indexed by time step
		obs_space=env.observation_space.shape[0]
		ac_space=env.action_space.n
		
		actions = []
		rewards = []
		states=[]
		actions_probs=[]
		is_t=False
		if(render):
			env.render()
		curr_state=np.expand_dims(env.reset(),axis=0)
		states.append(curr_state)

		while(not is_t):
			ac=self.model.predict(curr_state)
			ac=np.reshape(ac,[-1])
			curr_ac_prob=np.random.choice(ac,p=ac) #Action Probability
			curr_ac=np.where(ac==curr_ac_prob)[0][0] #Current Action
			actions_probs.append(curr_ac_prob)

			n_st,rew,is_t,_=env.step(curr_ac)
			n_st=np.expand_dims(n_st,axis=0) 
			
			states.append(n_st)
			actions.append(one_hot_vc(ac_space,curr_ac))
			rewards.append(rew)
			curr_state=np.copy(n_st)
		
		states=np.reshape(np.array(states),[-1,obs_space])
		actions=np.reshape(np.array(actions),[-1,ac_space]) #ONE_HOT_VECTOR

		return states[:-1], actions, rewards,np.expand_dims(np.array(actions_probs),axis=-1)

def parse_arguments():
	# Command-line flags are defined here.
	parser = argparse.ArgumentParser()
	parser.add_argument('--model-config-path', dest='model_config_path',
						type=str, default='LunarLander-v2-config.json',
						help="Path to the model config file.")
	parser.add_argument('--num-episodes', dest='num_episodes', type=int,
						default=50000, help="Number of episodes to train on.")
	parser.add_argument('--lr', dest='lr', type=float,
						default=5e-4, help="The learning rate.")

	# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
	parser_group = parser.add_mutually_exclusive_group(required=False)
	parser_group.add_argument('--render', dest='render',
							  action='store_true',
							  help="Whether to render the environment.")
	parser_group.add_argument('--no-render', dest='render',
							  action='store_false',
							  help="Whether to render the environment.")
	parser.set_defaults(render=False)

	return parser.parse_args()


def main(args):
	# Parse command-line arguments.
	args = parse_arguments()
	model_config_path = args.model_config_path
	num_episodes = args.num_episodes
	lr = args.lr
	render = args.render

	# Create the environment.
	env = gym.make('LunarLander-v2')
	# Load the policy model from file.
	with open(model_config_path, 'r') as f:
		model = keras.models.model_from_json(f.read())

	# TODO: Train the model using REINFORCE and plot the learning curve.
	agent=Reinforce(model,lr,env)
	agent.train(env)
if __name__ == '__main__':
	main(sys.argv)
