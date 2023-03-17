#!/usr/bin/env python3

from keras.layers import Dense, Activation, LeakyReLU
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import gym
import pickle
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import rospkg
import rospy
from active_vision.srv import controlSRV, controlSRVResponse, restartObjSRV, restartObjSRVResponse

class ReplayBuffer(object):
	def __init__(self, max_size, input_shape, n_actions, discrete=False):
		self.mem_size = max_size
		self.discrete = discrete
		self.state_memory = np.zeros((self.mem_size, input_shape))
		self.new_state_memory = np.zeros((self.mem_size, input_shape))
		if self.discrete:
			dtype = np.int8
		else:
			dtype = np.float32
		self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
		self.reward_memory = np.zeros(self.mem_size)
		self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
		self.mem_cntr = 0

	def store_transition(self, state, action, reward, new_state, done):
		index = self.mem_cntr % self.mem_size
		self.state_memory[index] = state
		self.new_state_memory[index] = new_state
		self.reward_memory[index] = reward
		self.terminal_memory[index] = 1 - int(done)
		if self.discrete:
			actions = np.zeros(self.action_memory.shape[1])
			actions[action] = 1.0
			self.action_memory[index] = actions
		else:
			self.action_memory[index] = action
		self.mem_cntr += 1

	def sample_buffer(self, batch_size):
		max_mem = min(self.mem_cntr, self.mem_size)
		batch = np.random.choice(max_mem, batch_size)
		states = self.state_memory[batch]
		new_states = self.new_state_memory[batch]
		rewards = self.reward_memory[batch]
		actions = self.action_memory[batch]
		terminals = self.terminal_memory[batch]
		return states, actions, rewards, new_states, terminals

def build_dqb(lr, n_actions, input_dims, fc1_dims, fc2_dims):
	model = Sequential([
			Dense(fc1_dims, input_shape=(input_dims, )),
			#LeakyReLU(),
			Activation('relu'),
			Dense(fc2_dims),
			#LeakyReLU(),
			Activation('relu'),
			Dense(fc2_dims),
			#LeakyReLU(),
			Activation('relu'),
			Dense(fc2_dims),
			#LeakyReLU(),
			Activation('relu'),
			Dense(n_actions)
		])
	model.compile(optimizer=Adam(lr=lr), loss='mse')
	return model

class Agent(object):
	def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_decount=0.996, epsilon_min=0.01, mem_size=100000, fname='q_model'):
		self.action_space = [i for i in range(n_actions)]
		self.n_actions = n_actions
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_decount = epsilon_decount
		self.epsilon_min = epsilon_min
		self.batch_size = batch_size
		base_dir = rospkg.RosPack().get_path('active_vision')
		path = base_dir + "/QLearning/NN_Models/"
		self.fname = path+fname


		self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
		self.q_eval = build_dqb(alpha, n_actions, input_dims, 128, 128)

	def remember(self, state, action, reward, new_state, done):
		self.memory.store_transition(state, action, reward, new_state, done)

	def choose_action(self, state):
		state = state[np.newaxis, :]
		rand = np.random.random()
		if rand < self.epsilon:
			action = np.random.choice(self.action_space)
		else:
			actions = self.q_eval.predict(state)
			action = np.argmax(actions)
		return action

	def learn(self):
		if self.memory.mem_cntr < self.batch_size:
			return
		state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
		action_values = np.array(self.action_space, dtype=np.int8)
		action_indices = np.dot(action, action_values)
		q_eval = self.q_eval.predict(state)
		q_next = self.q_eval.predict(new_state)

		q_target = q_eval.copy()
		batch_index = np.arange(self.batch_size, dtype=np.int32)

		q_target[batch_index, action_indices] = reward + self.gamma*np.max(q_next, axis=1)*done

		_ = self.q_eval.fit(state, q_target, verbose=0)

		if(self.epsilon > self.epsilon_min):
			self.epsilon = self.epsilon*self.epsilon_decount
		else:
			self.epsilon = self.epsilon_min

	def save_model(self, i):
		self.q_eval.save(self.fname+i+".h5")
		f = open(self.fname+i+"_mem", 'wb')
		pickle.dump(self.memory, f)

	def load_model(self, fname=None):
		if fname is None:
			target = self.fname
		else:
			target = fname
		self.q_eval = load_model(target+".h5")
		f = open(target+"_mem", 'rb')
		self.memory = pickle.load(f)

def display(epsilons, scores, avgs, stds):
	fig = plt.figure('F1')
	x = range(len(scores))
	fig.clear()
	fig.show()
	ax = fig.add_subplot(111)
	ax.plot(x,epsilons, 'r-', label='Random factor')
	ax.plot(x,scores, 'g-', label='Current Score')
	ax.plot(x,avgs, 'b-', label='Average Score')
	ax.plot(x,stds, 'm-', label='Standard Deviation')
	ax.legend()
	fig.canvas.draw()
	fig.canvas.flush_events()

if __name__ == '__main__':
	plt.ion()
	rospy.init_node('Q_controller')
	rospy.wait_for_service('/active_vision/restartEnv')
	rospy.wait_for_service('/active_vision/moveKinect')
	restartEnv = rospy.ServiceProxy('/active_vision/restartEnv', restartObjSRV)
	nextMove = rospy.ServiceProxy('/active_vision/moveKinect', controlSRV)
	base_dir = rospkg.RosPack().get_path('active_vision')
	path = base_dir + "/QLearning/NN_Models/"
	#env = gym.make('LunarLander-v2')
	n_games = 2000
	grid_size = 5  # Size of the state vector
	num_inputs = (grid_size * grid_size)*2+2
	num_actions = 8
	objs = [1, 2, 3, 5, 6, 7]
	poses = [1, 2, 3, 1, 1, 1] #2
	yaws = [89, 179, 179, 89, 359, 179] #179
	convergence_target = 0.7

	agent = Agent(alpha=0.005, gamma=0.9, n_actions=num_actions, epsilon=1.0, batch_size=64, input_dims=num_inputs, epsilon_decount=0.996, epsilon_min=0.01, mem_size=200000, fname='4layersall2_')
	scores = []
	std_devs = []
	avg_scores = []
	eps_history = []
	agent.load_model(path+'4layersall300')
	best = 0

	for i in range(n_games):
		done = False
		score = 0
		cStep = 0
		c = np.random.choice(len(objs))
		obj = objs[c]
		pose = np.random.choice(poses[c])
		yaw = np.random.choice(yaws[c])
		#print("Testing object %d, pose %d, yaw %d" % (obj, pose, yaw))
		observation = np.array(restartEnv(obj, pose, yaw).stateVec.data)
		#observation = env.reset()
		while not done and cStep <= 5:
			#print(cStep)
			#env.render()
			action = agent.choose_action(observation)
			#new_observation, reward, done, info = env.step(action)
			ret = nextMove(action+1)
			new_observation = np.array(ret.stateVec.data)
			done = ret.done
			cStep = ret.steps
			if(done):
				reward = 5
			else:
				reward = -1
			agent.remember(observation, action, reward, new_observation, done)
			observation = new_observation
			agent.learn()
			score += reward
		if not done and 5 == cStep:
			done = True

		eps_history.append(agent.epsilon*5)
		scores.append(score)
		avg_score = np.mean(scores[max(0, i-100):(i+1)])
		if avg_score > best and i >= 100:
			best = avg_score
		avg_scores.append(avg_score)
		std = np.std(scores[max(0, i-100):(i+1)])
		std_devs.append(std)

		print('obj (0-5) ', c, ' episode ', i, ' score %.2f ' %score, ' average score %.2f ' %avg_score, ' best score %.2f ' %best, ' std_dev %.2f' %std)
		if i % 10 == 0 and i > 0:
			agent.save_model("")
			if std < convergence_target:
				print("Reached target!")
				break
		if i % 100 == 0 and i > 0:
			agent.save_model(str(i))
		if i % 1 == 0 and i > 0:
			display(eps_history, scores, avg_scores, std_devs)