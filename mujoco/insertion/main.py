
from SAC_env import SAC_Agent
from common.utils import mini_batch_train
import gym

env = gym.make("Pendulum-v0")




gamma = 0.99
tau = 0.01
alpha = 0.2
a_lr = 3e-4
q_lr = 3e-4
p_lr = 3e-4
buffer_maxlen = 1000000

state = env.reset()


#2019 agent
agent = SAC_Agent(env, gamma, tau, alpha, q_lr, p_lr, a_lr, buffer_maxlen, 32)

# train
episode_rewards = mini_batch_train(env, agent, 50, 500, 64)
