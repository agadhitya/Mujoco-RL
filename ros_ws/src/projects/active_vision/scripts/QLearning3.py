#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
import gym
import sys, signal, time
import matplotlib.pyplot as plt

class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity-1)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = int((tree_index - 1)/2)
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0
        while True:
            left_child_index = 2*parent_index + 1
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]
    

class Memory(object):
    PER_e = 0.01
    PER_a = 0.6
    PER_b = 0.4
    PER_b_increment_per_sampling = 0.001
    absolute_error_upper = 1.0

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, experience):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        self.tree.add(max_priority, experience)

    def sample(self, n):
        memory_b = []
        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)
        priority_segment = self.tree.total_priority/n
        self.PER_b = np.min([1.0, self.PER_b+self.PER_b_increment_per_sampling])

        p_min = np.min(self.tree.tree[-self.tree.capacity:])/self.tree.total_priority
        max_weight = (p_min*n)**(-self.PER_b)

        for i in range(n):
            a, b = priority_segment*i, priority_segment*(i+1)
            value = np.random.uniform(a, b)

            index, priority, data = self.tree.get_leaf(value)
            sampling_probabilities = priority/self.tree.total_priority

            b_ISWeights[i, 0] = np.power(n*sampling_probabilities, -self.PER_b)/ max_weight
            b_idx[i] = index
            experience = [data]
            memory_b.append(experience)
        return b_idx, memory_b, b_ISWeights

    def batch_update(self, tree_index, abs_errors):
        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)
        for ti, p in zip(tree_index, ps):
            self.tree.update(ti, p)

class DisplayData(object):
    max_x = 300
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.c_R = []
        self.running_R = []
        self.max_R = []
        self.ax.plot([-1],[0], 'r-', [-1],[0], 'g-', [-1],[0], 'b-')

    def update(self, x, c_R, running_R, max_R):
        c_min = np.maximum((x-self.max_x), 0)
        self.c_R.append(c_R)
        self.running_R.append(running_R)
        self.max_R.append(max_R)
        if c_min > 0:
            self.c_R = self.c_R[1:]
            self.running_R = self.running_R[1:]
            self.max_R = self.max_R[1:]
        x_c = range(c_min, x+1, 1)
        self.ax.plot(x_c, self.c_R, 'r-', x_c, self.running_R, 'g-', x_c, self.max_R, 'b-')
        self.fig.canvas.draw()
        plt.cla()

plt.ion()
path = '/home/diyogon/Downloads/tempModel'
# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000

env = gym.make("CartPole-v1")
env.seed(seed)
eps = np.finfo(np.float32).eps.item()
print(env.action_space)

num_inputs = 4
num_actions = 2

model = None
model_target = None

def keyboardInterruptHandler(signal, frame):
    print("Caught interrupt, saving models")
    save_model()
    print("Models saved.")
    sys.exit(0)

def build_model():
    inputs = layers.Input(shape=(num_inputs,))
    common = layers.Dense(20, activation="relu")(inputs)
    common1 = layers.Dense(20, activation="relu")(common)
    action = layers.Dense(num_actions, activation="softmax")(common1)
    common2 = layers.Dense(20, activation="relu")(common)
    value = layers.Dense(1, activation=None)(common2)
    output = value + tf.subtract(action, tf.reduce_mean(action, axis=1, keepdims=True))
    return keras.Model(inputs=inputs, outputs=output)

def save_model():
    model.save(path)
    model_target.save(path+"_target")

model = build_model()
model_target = build_model()
# model.save(path)
# model_target.save(path+"_target")
#model = tf.keras.models.load_model(path)
#model_target = tf.keras.models.load_model(path+"_target")

optimizer = keras.optimizers.Adam(learning_rate=0.00025)

disp = DisplayData()

action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
max_reward = 0
running_reward = 0
episode_count = 0
frame_count = 0
# Number of frames to take random action and observe output
epsilon_random_frames = 1000
# Number of frames for exploration
epsilon_greedy_frames = 100000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 10
# How often to update the target network
update_target_network = 10000
# How often to display data
update_human = 100
# Using huber loss for stability
loss_function = keras.losses.Huber()

memory = Memory(max_memory_length)

print("Starting")
while True:  # Run until solved
    state = np.array(env.reset())
    episode_reward = 0
    steps = 0

    for timestep in range(1, max_steps_per_episode):
        #env.render(); #Adding this line would show the attempts
        # of the agent in a pop up window.
        frame_count += 1
        steps += 1

        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Normalize because numpy whines otherwise
            #action_probs = tf.cast(action_probs, tf.float64)
            #action_probs = tf.compat.v1.math.softmax(action_probs)
            # Take weighted action
            #action = np.random.choice(range(action_probs.shape[1]), p=action_probs[0])
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        state_next, reward, done, _ = env.step(action)
        state_next = np.array(state_next)

        episode_reward += reward

        # Save actions and states in replay buffer
        experience = state, action, reward, state_next, done
        memory.store(experience)
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            print("**********************")
            template = "Updated Network with running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode_count, frame_count))
            save_model()


        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            if episode_reward > max_reward:
                max_reward = episode_reward
            # print("**********************")
            # print("Episode: %d" % episode_count)
            # print("Reward: %.1f" % episode_reward)
            # print("Running reward: %.1f" % running_reward)
            # print("Max reward: %.1f" % max_reward)
            disp.update(episode_count, episode_reward, running_reward, max_reward)
            break

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1

    if running_reward > 400:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        model.save(path)
        model_target.save(path+"_target")
        break