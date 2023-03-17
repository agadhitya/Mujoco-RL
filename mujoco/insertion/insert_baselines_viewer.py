import robosuite as suite
from robosuite import load_controller_config
import numpy as np
from robosuite.wrappers import GymWrapper
from gym import spaces
from stable_baselines3 import PPO

np.seterr(divide='ignore', invalid='ignore')

class InsertionViewer:
    def __init__(self,num_episodes=500, max_t=300,
                 weight_name=None, show_sim=True, verbose=True):
        '''Initialize Insertion Trainer using Mujoco and Robosuite.

        num_episodes: Number of episodes for viewing.
        max_t: Number of timesteps per episode.
        show_sim: Whether to show graphical simulation
        verbose: Whether to print useful output.
        '''

        self.num_episodes=num_episodes
        self.max_t=max_t
        self.control='OSC_POSE'
        self.show_sim=show_sim
        self.verbose=verbose
        self.episode_rewards=[]
        self.control_dof=7 #XYZ RPY Gripper
        self.weight_name=weight_name

        # Configure robosuite enviroment
        # Load the desired controller's default config as a dict
        self.config = load_controller_config(default_controller=self.control)

                # Create environment
        self.env =GymWrapper(
                    suite.make("Insert",
                                robots="Panda",
                                controller_configs=self.config,
                                has_renderer=self.show_sim,
                                has_offscreen_renderer=False,
                                use_camera_obs=False,
                                ),
                    keys=['robot0_eef_pos',  #3 XYZ
                          #'robot0_eef_quat', #4 EE Quaternion
                          'ft_sensor',       #6 fx,fy,fz,tx,ty,tz
                          ]
        )

        #Define action space for OSC_POSE control
        self.env.action_space = spaces.Box(low=np.array([-0.1,-0.1,-0.1,0.0,0.0,0.0,0.0]),
                                           high=np.array([0.1,0.1,0.1,0.0,0.0,0.0,0.0]))
        # self.env.observation_space = spaces.Box(low=np.array([-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.]),
                                                # high=np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.]))
        #Pass episode time limit to Insert environment.
        self.env.env.horizon=self.max_t
        #Get panda object
        self.panda=self.env.robots[0]
        #Create agent
        self.agent = PPO("MlpPolicy", self.env, verbose=1)

    def view_sim(self):
        '''Runs Peg Insertion simulations.'''
        self.agent.load(self.weight_name, env=self.env)
        for i_episode in range(self.num_episodes):
            if self.verbose:
                print(f'Episode {i_episode + 1}/{self.num_episodes}')
            obs = self.env.reset()
            episode_reward = 0

            for t in range(self.max_t):
                action, _states = self.agent.predict(obs)

                obs, reward, done, info = self.env.step(action)

                episode_reward += reward

                if self.show_sim:
                    self.env.render()

                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break

            if self.verbose:
                print(episode_reward)

            self.episode_rewards.append(episode_reward)
        self.env.close()

if __name__=='__main__':
    it=InsertionViewer(num_episodes=5, max_t=4096,
                       weight_name="4096000steps_05102021_1704", #For loading weights
                       show_sim=True, verbose=True)
    it.view_sim()
