import robosuite as suite
from robosuite import load_controller_config
import numpy as np
#import roboticstoolbox as rtb
#from spatialmath import *
from robosuite.wrappers import GymWrapper
from SAC_env import SAC_Agent
import matplotlib.pyplot as plt
from gym import spaces
import pandas as pd

#TODO: Pass self.max_t through to insertion_env (maxes out at 1000)

#-----
#TODO: Add EE orientation to reward function and action space
#TODO: Add proper parameters (friction, inertia, density, dimensions, etc. to all objects. See insert.py)
#TODO: Understand why ee0 initialize isn't working. Currently hardcoded in panda_robot.py
#TODO: Turn off 'reward_shaping' in insert.py if desired

#TODO: Identify coordinate frames (peg and hole).
#TODO: Understand why check_for_success is not working properly.

class InsertionTrainer:
    def __init__(self,num_episodes=500, max_t=300, ee0=[0.4, 0.0, 0.3],
                 action_type='random', control="JOINT_POSITION", filename='results',
                 weight_basename=None, show_sim=True, verbose=True):
        '''Initialize Insertion Trainer using Mujoco and Robosuite.

        num_episodes: Number of episodes for training.
        max_t: Number of timesteps per episode.
        ee0: Starting EE point [x,y,z] in meters. [TODO: Make this argument actually work.]
        action_type: sac, random, nothing, or down (down for OSC_POSE only).
        control: Control type. Choose from: OSC_POSE or JOINT_POSITION.
        show_sim: Whether to show graphical simulation
        verbose: Whether to print useful output.
        '''

        self.num_episodes=num_episodes
        self.max_t=max_t
        self.ee0=ee0
        self.action_type=action_type
        self.control=control
        self.home=None
        self.show_sim=show_sim
        self.verbose=verbose
        self.episode_rewards=[]
        self.filename=filename #f'{num_episodes}ep_{max_t}maxt_'

        # Configure robosuite enviroment
        # Load the desired controller's default config as a dict
        self.config = load_controller_config(default_controller=control)

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
                          #'robot0_joint_pos_cos',
                          #'robot0_joint_pos_sin',
                          #'robot0_joint_vel',
                          #'robot0_gripper_qpos',
                          #'robot0_gripper_qvel',
                          #'hole_pos',
                          #'hole_quat',
                          #'gripper_to_hole_pos',
                          #'robot0_proprio-state',
                          #'object-state',
                          ]
        )
        #Define action space for OSC_POSE control
        self.env.action_space = spaces.Box(low=np.array([-0.1,-0.1,-0.1,0.0,0.0,0.0,0.0]), 
                                           high=np.array([0.1,0.1,0.01,0.0,0.0,0.0,0.0]))

        #Define Panda model for analytical calculations
        #self.rtb_panda=rtb.models.DH.Panda()
        #Calculate home pose for given control type. Currently commented
        #out since it is hardcorded in panda object.
        #self.get_start_pose()

        #Get panda object
        self.panda=self.env.robots[0]
        #self.panda.init_qpos=self.home
        # self.panda.set_robot_joint_positions(self.home)

        #Find length of control space vector
        if self.control=='JOINT_POSITION': #Joints1-7 Gripper
            self.control_dof=8
        elif self.control=='OSC_POSE': #XYZ RPY Gripper
            self.control_dof=7
        else:
            raise("control_dof unknown for given control type.")

        #Load SAC params. Create agent.
        if self.action_type.lower()=='sac':
            # Define SAC params
            self.gamma = 0.99
            self.tau = 0.01
            self.alpha = 0.2
            self.a_lr = 3e-4
            self.q_lr = 3e-4
            self.p_lr = 3e-4
            self.buffer_maxlen = 1000000
            self.batch_size = 100
            #Create SAC agent
            self.agent = SAC_Agent(self.env,
                                   self.gamma,
                                   self.tau,
                                   self.alpha,
                                   self.q_lr,
                                   self.p_lr,
                                   self.a_lr,
                                   self.buffer_maxlen,
                                   self.batch_size,
                                   weight_basename=weight_basename)

    def get_action(self,obs):
        '''Create action vector based on initial input.'''
        if self.action_type.lower()=='random':
            return np.random.randn(self.control_dof)
        elif self.action_type.lower()=='nothing':
            return np.zeros(self.control_dof)
        elif self.action_type.lower()=='down':
            if self.control=='OSC_POSE': #XYZ RPY Gripper
                return np.array([0,0,-0.025,0,0,0,0.5])
            else:
                raise("control_dof unknown for given control type.")
        elif self.action_type.lower()=='sac':
            #Get update from SAC
            action=self.agent.get_action(obs)
            #print(action_z)
            #return np.array([0,0,action[2],0,0,0,0])
            #return np.array([action[0],action[1],-0.05,0,0,0,0])
            #return np.array([action[0],action[1],action[2],0,0,0,0])
            return action
        else:
            raise('Undefined action type.')

    # def get_start_pose(self):
    #     '''Calculate starting pose for given control type.'''
    #     if self.ee0=='start':
    #         self.home=np.array([0.0,-0.785,0.0,-2.356, 0.0, 1.57, 0.784])
    #     else:
    #         #Use IK from Peter Corke's Toolbox. OA defines orientation.
    #         T = SE3(self.ee0[0],self.ee0[1],self.ee0[2])*SE3.OA([0, -1, 0], [0, 0, -1])
    #         joint_space_sol=self.rtb_panda.ikine_min(T)
    #         print(joint_space_sol)

    #         self.home=np.array(joint_space_sol.q)

    def plot(self):
        '''Plots episode rewards.'''
        plt.plot(self.episode_rewards,'o-')
        plt.grid()
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'Rewards per Episode. Batch: {self.batch_size}, Max. t: {self.max_t}')
        plt.savefig(self.filename+'.png')
        #plt.show()

    def save_rewards(self):
        '''Saves rewards from each episode to xlsx format.'''
        df = pd.DataFrame(self.episode_rewards)
        writer = pd.ExcelWriter(self.filename+'.xlsx', engine='xlsxwriter')
        df.to_excel(writer, sheet_name='trial1', index=False)
        writer.save()
        print('Saved results to '+self.filename+'.xlsx')

    def run_sim(self):
        '''Runs Peg Insertion simulations.'''
        for i_episode in range(self.num_episodes):
            if self.verbose:
                print(f'Episode {i_episode + 1}/{self.num_episodes}')
            obs = self.env.reset()
            episode_reward = 0

            for t in range(self.max_t):
                action = self.get_action(obs)
                next_state, reward, done, info = self.env.step(action)

                #Hack for setting "done". [How can this be properly done in insert.py?]
                if reward>0:
                    done=True
                    #Hack for scaling rewards to reduce time required
                    #reward*=((self.max_t-t)/self.max_t) #linear decay
                    reward*=(0.02**(t/self.max_t)) #exp. decay
                    #TODO: implement exp decay with negative rewards after a while

                episode_reward += reward
                #Getting peg/hole poses in world frame
                if self.action_type.lower()=='sac':
                    self.agent.replay_buffer.push(obs, action, reward, next_state, done)
                    obs = next_state

                # if t%50==0:
                #     print('Peg')
                #     peg_pos_in_world = self.env.sim.data.get_body_xpos(self.env.peg.root_body)
                #     peg_rot_in_world = self.env.sim.data.get_body_xmat(self.env.peg.root_body).reshape((3, 3))
                #     print(peg_pos_in_world)
                #     #print(peg_rot_in_world)
                #     print('Hole')
                #     hole_pos_in_world = self.env.sim.data.get_body_xpos(self.env.hole.root_body)
                #     hole_rot_in_world = self.env.sim.data.get_body_xmat(self.env.hole.root_body).reshape((3, 3))
                #     print(hole_pos_in_world)

                #     print('*********')

                if self.show_sim:
                    self.env.render()

                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break

            #After each episode:
            if self.action_type.lower()=='sac':
                self.agent.update(self.batch_size)

            if self.verbose:
                print(episode_reward)

            self.episode_rewards.append(episode_reward)
        self.env.close()
        self.save_rewards()

        self.agent.save(self.filename)


if __name__=='__main__':
    it=InsertionTrainer(num_episodes=10, max_t=5000, ee0=[0.67,0.0,0.38], #(0.6375,0.0,0.45)
                        action_type='sac', control="OSC_POSE",
                        filename='loaded_test',
                        weight_basename='500ep_5000maxt_expreward_noOrientObs',
                        show_sim=True, verbose=True)
    it.run_sim()
    it.plot()

    #OSC/World:
        #X is coming towards you.
        #Y is to the right.
        #Z is up.

    #Peg:
        #Z is roughly aligned with Hole.

    #Hole (XY Origin as at center of table):
        #Z is roughly aligned with Peg.