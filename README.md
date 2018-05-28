Implementations of RL algorithms-
- DQN
- Advantage Actor-Critic
- Imitation Learning
- REINFORCE
- Dueling DQN

| CartPole                        | Mountain Car                          | LunarLander                             | 
| ------------------------------- | ------------------------------------- | --------------------------------------- | 
| ![CartPole](/docs/CartPole.gif) | ![MountainCar](/docs/MountainCar.gif) | ![LunarLander](/docs/LunarLander-2.gif) |


Requirements-
 - TensorFlow
 - Keras
 - Gym Box2D envs

To run DQN and Dueling DQN - <br />
`python DQN_Implementation.py` with the following arguments- <br />

Argument | Description
--- | --- 
`--env=ENVIRONMENT_NAME`| CartPole-v0, MountainCar-v0, LunarLander-v0 
`--render=1 OR 0` | variable to enable render(1) or not(0)
`--train=1 OR 0` |  variable to train(1) the model or not(0) 
`--type=MODEL_TYPE` | DQN,Dueling
`--save_folder=FOLDER_DIR`| folder directory to save videos (Optional). Videos are not saved if nothing is given
`--model_file=FILE_DIR` | File directory of saved model(Optional). Nothing is done if not given    

HyperParameters have been sectioned for easy alteration. You should be able to locate them easily by just searching 'Hyper'.

To run - 
 - Advantage-Actor Critic - `python a2c.py` 
 - REINFORCE - `python reinforce.py --render`
 - Imitation - `python imitation --render` 