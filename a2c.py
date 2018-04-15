import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym
import keras.backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gym.wrappers import Monitor
from reinforce import Reinforce
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,Input,Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import os
from collections import deque
import pickle
from keras.utils import multi_gpu_model

_EPSILON=1e-7
def plot_this(num_episodes,mean_,std_,his_len,n,save_folder):
    x=np.linspace(0,num_episodes,num_episodes/his_len+1)
    plt.errorbar(x, mean_, yerr=std_,linestyle=None,marker='^')
    plt.savefig(os.path.join(save_folder,"Plot_N{}_a2c".format(n)))
    plt.show()


def actor_loss(y_true,y_pred):
    ''' Actor Loss'''
    #y_true=[0,0,R_t-V_w,0]
    #y_pred=[probailitiy distributions]
    epsilon=tf.convert_to_tensor(_EPSILON,y_pred.dtype.base_dtype)
    probable_acs=tf.clip_by_value(y_pred,epsilon,1-epsilon)
    # entropy=tf.multiply(probable_acs,tf.log(probable_acs))
    # entrpy_loss=tf.reduce_mean(tf.reduce_sum(entropy,axis=-1))
    loss=tf.multiply(y_true,tf.log(probable_acs))#[0*log(P(0|S),0*log(P(1|S),(R_t-V_w)*log(P(2|S)),0*log(P(3|S))]
    loss_vector=K.sum(loss,axis=1) #0+0+G_t*P(2|S)+0 = G_t*P(2|S)
    loss=tf.reduce_mean(loss_vector)#self explanatory
    return -loss

def critic_loss(y_true,y_pred):
    loss=tf.reduce_mean(tf.square(y_true-y_pred))
    return loss

class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class.
    def __init__(self, model, lr, critic_model, critic_lr,save_folder, n=20):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        # - save_folder:Directoryo to save model files and videos recorded
        self.model = model
        self.critic_model = critic_model
        self.n = n
        self.save_folder=save_folder
        # self.model = multi_gpu_model(model, gpus=2)
        # self.critic_model = multi_gpu_model(critic_model, gpus=2)
        self.model.summary()
        self.critic_model.summary()
        self.model.compile(loss=actor_loss,optimizer=Adam(lr=lr))
        self.critic_model.compile(loss=critic_loss,optimizer=Adam(lr=critic_lr))

    def save_model(self,itr):
        self.model.save(os.path.join(self.save_folder, "actor-n{}-{}.h5".format(self.n,itr)))
        self.critic_model.save(os.path.join(self.save_folder,"critic-n{}-{}.h5".format(self.n,itr)))
       
    def load_model(self, actor_model_file,critic_model_file):
        # Helper function to load an existing model.
        self.model=keras.models.load_model(actor_model_file, custom_objects={"K": K,"actor_loss" : actor_loss,"critic_loss":critic_loss})
        self.critic_model=keras.models.load_model(critic_model_file, custom_objects={"K": K,"actor_loss" : actor_loss,"critic_loss":critic_loss})

    def load_model_weights(self,actor_weight_file,critic_weight_file):
        # Helper funciton to load model weights. 
        self.model.load_weights(weight_file)
        self.critic_model.load_weights(critic_weight_file)

    def train(self, env,num_episodes=50000, gamma=1.0,save_videos=0,update_ac=1):
        # Trains the model on a single episode using A2C.
        his_len=200
        rew_history=deque(maxlen=his_len)
        step_history=deque(maxlen=his_len)
        render=False
        if(save_videos==1):
            env=Monitor(env, os.path.join(self.save_folder,"videos"),video_callable=lambda episode_id:episode_id%500==0,force=True)
        mean_total=[]
        std_total=[]

        for l in range(num_episodes+1):
            states,actions_OH,rewards,_=self.generate_episode(env,render)
            R_total=[]
            total_T=len(rewards)

            for i in range(total_T-1,-1,-1):
                v_end=0 if((i+self.n)>=total_T) else self.critic_model.predict(np.expand_dims(states[i],axis=0))
                r_t=0
                for k in range(self.n):
                    r_t=r_t+(gamma**k)*rewards[i+k] if((i+k)<total_T) else r_t+0
                R_t=(gamma**self.n)*v_end+r_t
                R_total.append(R_t)
            R_total=np.flip(np.array(R_total),axis=0)
            R_total=np.expand_dims(R_total,axis=-1)
            V_w_epi=self.critic_model.predict(states)
            R_total_actions=np.multiply(R_total-V_w_epi,actions_OH)

            critic_y_true=R_total-V_w_epi
            if(l%update_ac==0):
                history_ac=self.model.fit(states,R_total_actions*1e-2,verbose=0,epochs=1,batch_size=len(states))
            history_critic=self.critic_model.fit(states,critic_y_true*1e-1,verbose=0,epochs=1,batch_size=len(states))
            
            if(len(rew_history)>=his_len):
                rew_history.popleft()
            if(len(step_history)>=his_len):
                step_history.popleft()
            step_history.append(len(states))
            rew_history.append(np.sum(np.array(rewards)))
            
            if(l%300==0):
                mean_rew,mean_steps=0,0 
                for i in range(len(rew_history)):
                    mean_rew+=rew_history[i]
                    mean_steps+=step_history[i]
                mean_steps/=len(step_history)
                mean_rew/=len(rew_history)
                mean_100,std_100=self.test(env)
                mean_total.append(mean_100)
                std_total.append(std_100)
              
                print("##########TEST####################################")
                print("Mean TEST={},STD_DEV_TEST={}".format(mean_100,std_100))
                print("###################################################")
                print("Mean Rew={},Mean_steps={},Loss_ac={},Loss_critic={} for epi={}".format(mean_rew,mean_steps,history_ac.history['loss'],history_critic.history['loss'],l))
                self.save_model(l)
        mean_total=np.expand_dims(np.array(mean_total),axis=-1)
        std_total=np.expand_dims(np.array(std_total),axis=-1)
        pickle.dump(mean_total,open(os.path.join(self.save_folder,"mean_total_a2c-N_{}.pkl".format(self.n)),"wb"))
        pickle.dump(std_total,open(os.path.join(self.save_folder,"std_total_N_a2c-{}.pkl".format(self.n)),"wb"))

        plot_this(num_episodes,mean_total,std_total,his_len,self.n,self.save_folder)
        
        return

# def make_critic_model(env):
#     model=Sequential()
#     model.add(Dense(16,input_dim=env.observation_space.shape[0],kernel_initializer='he_uniform'))
#     model.add(Activation('relu'))
#     # model.add(BatchNormalization())
#     model.add(Dense(32,kernel_initializer='he_uniform'))
#     model.add(Activation('relu'))
#     # model.add(BatchNormalization())
#     # model.add(Dense(16,kernel_initializer='he_uniform'))
#     # model.add(Activation('relu'))
#     model.add(Dense(1,activation='linear',kernel_initializer='he_uniform'))
#     return model

def define_models(env):
	obs_input=Input(shape=(env.observation_space.shape[0],))
	common_model=Dense(16,activation='relu')(obs_input)
	common_model=Dense(16,activation='relu')(common_model)
	common_model=Dense(16,activation='relu')(common_model)
	actor_pred=Dense(env.action_space.n,activation='softmax')(common_model)
	actor_model=Model(inputs=obs_input,outputs=actor_pred)
	# critic_layer=Lambda(lambda x: K.stop_gradient(x))(common_model)
	critic_layer=Dense(32,activation='linear')(common_model)
	critic_layer=Dense(16,activation='linear')(critic_layer)
	critic_pred=Dense(1,activation='linear')(critic_layer)	
	critic_model=Model(inputs=obs_input,outputs=critic_pred)
	
	return actor_model,critic_model

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the actor model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")
    parser.add_argument('--save_folder',dest='save_folder',type=str,
                        default=os.getcwd(),help="Directory to save model videos")
    parser.add_argument('--actor_model_file',dest='actor_model_file',type=str,
                    default=None,help="Actor Model File")  
    parser.add_argument('--critic_model_file',dest='critic_model_file',type=str,
                    default=None,help="Critic model file")
    parser.add_argument('--save_videos',dest='save_videos',type=int,
                    default=0,help="Whether to save videos or not")
    parser.add_argument('--update_ac',dest='update_ac',type=int,
                    default=1,help="Update freq of actor")
    parser.add_argument('--gamma',dest='gamma',type=float,
                    default=1.0,help="Discount Factor")
    
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
    critic_lr = args.critic_lr
    n = args.n
    render = args.render

    print("Configuration: \n","N= {}, Ac Lr={}, Cr_lr={} \n".format(n, lr,critic_lr),
        "Models saving in {}".format(args.save_folder))

    # Create the environment.
    env = gym.make('LunarLander-v2')
    
    # Load the actor model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())
    # actor_model,critic_model=make_critic_model(env)
	actor_model,critic_model=define_models(env)

    agent=A2C(actor_model, lr, critic_model, critic_lr,args.save_folder,n)

    if(args.actor_model_file is not None and args.critic_model_file is not None):
        agent.load_model(args.actor_model_file,args.critic_model_file)
    
    agent.train(env,num_episodes=num_episodes,gamma=args.gamma,save_videos=args.save_videos,update_ac=args.update_ac)


if __name__ == '__main__':
    main(sys.argv)
