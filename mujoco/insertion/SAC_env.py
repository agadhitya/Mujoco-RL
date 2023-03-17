import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from SAC import Value_Network,SoftQ_Network,Policy_Network
from common.replay_buffers import BasicBuffer

class SAC_Agent:

    def __init__(self, env, gamma, tau, alpha, q_lr, policy_lr,a_lr,
                 buffer_maxlen,batch_size, weight_basename=None):

        device = "cuda"

        self.device = torch.device(device)
        print('Using CUDA Device: '+torch.cuda.get_device_name(0))

        self.env = env
        self.action_range = [env.action_space.low,env.action_space.high]
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.batch_size = batch_size

        self.gamma = gamma
        self.tau = tau
        self.update_step = 0
        self.delay_step = 2

        self.q_net1 = SoftQ_Network(self.obs_dim, self.action_dim).to(self.device)
        self.q_net2 = SoftQ_Network(self.obs_dim, self.action_dim).to(self.device)
        self.target_q_net1 = SoftQ_Network(self.obs_dim, self.action_dim).to(self.device)
        self.target_q_net2 = SoftQ_Network(self.obs_dim, self.action_dim).to(self.device)
        self.policy_net = Policy_Network(self.obs_dim, self.action_dim).to(self.device)

        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(param)

        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(param)

        # initialize optimizers
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)


        self.alpha = alpha
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=a_lr)

        self.replay_buffer = BasicBuffer(buffer_maxlen)

        if weight_basename is not None:
            self.weight_basename=weight_basename
            self.load()

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.policy_net.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = action.cpu().detach().squeeze(0).numpy()

        return self.rescale_action(action)

    def rescale_action(self, action):
        return action * (self.action_range[1] - self.action_range[0]) / 2.0 +\
            (self.action_range[1] + self.action_range[0]) / 2.0

    def update(self, batch_size):
        # print(batch_size)
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        dones = dones.view(dones.size(0), -1)

        next_actions, next_log_pi = self.policy_net.sample(next_states)
        next_q1 = self.target_q_net1(next_states, next_actions)
        next_q2 = self.target_q_net2(next_states, next_actions)
        next_q_target = torch.min(next_q1, next_q2) - self.alpha * next_log_pi
        expected_q = rewards + (1 - dones) * self.gamma * next_q_target

        # q loss
        curr_q1 = self.q_net1.forward(states, actions)
        curr_q2 = self.q_net2.forward(states, actions)
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        # update q networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # delayed update for policy network and target q networks
        new_actions, log_pi = self.policy_net.sample(states)
        if self.update_step % self.delay_step == 0:
            min_q = torch.min(
                self.q_net1.forward(states, new_actions),
                self.q_net2.forward(states, new_actions)
            )
            policy_loss = (self.alpha * log_pi - min_q).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # target networks
            for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # update temperature
        alpha_loss = (self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        self.update_step += 1

        return next_states

    def save(self, base_filename):
        '''Saves state_dict for each network.'''
        models={'q_net1':self.q_net1,
                'q_net2':self.q_net2,
                'target_q_net1':self.target_q_net1,
                'target_q_net2':self.target_q_net2,
                'policy_net':self.policy_net}

        for name, model in models.items():
            torch.save(model.state_dict(), base_filename+'_'+name+'.pt')

        print('Saved all networks.')

    def load(self):
        '''Loads state_dicts for each network.'''
        models={'q_net1':self.q_net1,
                'q_net2':self.q_net2,
                'target_q_net1':self.target_q_net1,
                'target_q_net2':self.target_q_net2,
                'policy_net':self.policy_net}

        for name, model in models.items():
            model.load_state_dict(torch.load(self.weight_basename+'_'+name+'.pt'))

        print('Loaded all networks.')