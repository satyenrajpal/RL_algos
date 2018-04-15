Implementations of RL algorithms-
- DQN
- Advantage Actor-Critic
- Imitation Learning
- REINFORCE
- Dueling DQN

| CartPole                        | Mountain Car                          | LunarLander                           | 
| ------------------------------- | ------------------------------------- | ------------------------------------- | 
| ![CartPole](/docs/CartPole.gif) | ![MountainCar](/docs/MountainCar.gif) | ![LunarLander](/docs/LunarLander.gif) |


Requirements-
 - TensorFlow
 - Keras
 - Gym Box2D envs

To run the program run- <br />
`python DQN_Implementation.py` with the following arguments- <br />

Argument | Description
--- | --- 
`--env=ENVIRONMENT_NAME`| CartPole-v0, MountainCar-v0 
`--render=1 OR 0` | variable to enable render(1) or not(0)
`--train=1 OR 0` |  variable to train(1) the model or not(0) 
`--type=MODEL_TYPE` | DQN,Dueling
`--save_folder=FOLDER_DIR`| folder directory to save videos (Optional). Videos are not saved if nothing is given
`--model_file=FILE_DIR` | File directory of saved model(Optional). Nothing is done if not given    

HyperParameters have been sectioned for easy alteration. You should be able to locate them easily by just searching 'Hyper' and alter them as per your convenience.
