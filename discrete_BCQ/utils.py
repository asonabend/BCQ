import cv2
import gym
import numpy as np
import torch

def ReplayBuffer(state_dim, is_atari, atari_preprocessing, batch_size, buffer_size, device):
	if is_atari: 
		return AtariBuffer(state_dim, atari_preprocessing, batch_size, buffer_size, device)
	else: 
		return StandardBuffer(state_dim, batch_size, buffer_size, device)


class AtariBuffer(object):
	def __init__(self, state_dim, atari_preprocessing, batch_size, buffer_size, device):
		self.batch_size = batch_size
		self.max_size = int(buffer_size)
		self.device = device

		self.state_history = atari_preprocessing["state_history"]

		self.ptr = 0
		self.crt_size = 0

		self.state = np.zeros((
			self.max_size + 1,
			atari_preprocessing["frame_size"],
			atari_preprocessing["frame_size"]
		), dtype=np.uint8)

		self.action = np.zeros((self.max_size, 1), dtype=np.int64)
		self.reward = np.zeros((self.max_size, 1))
		
		# not_done only consider "done" if episode terminates due to failure condition
		# if episode terminates due to timelimit, the transition is not added to the buffer
		self.not_done = np.zeros((self.max_size, 1))
		self.first_timestep = np.zeros(self.max_size, dtype=np.uint8)


	def add(self, state, action, next_state, reward, done, env_done, first_timestep):
		# If dones don't match, env has reset due to timelimit
		# and we don't add the transition to the buffer
		if done != env_done:
			return

		self.state[self.ptr] = state[0]
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		self.first_timestep[self.ptr] = first_timestep

		self.ptr = (self.ptr + 1) % self.max_size
		self.crt_size = min(self.crt_size + 1, self.max_size)


	def sample(self):
		ind = np.random.randint(0, self.crt_size, size=self.batch_size)

		# Note + is concatenate here
		state = np.zeros(((self.batch_size, self.state_history) + self.state.shape[1:]), dtype=np.uint8)
		next_state = np.array(state)

		state_not_done = 1.
		next_not_done = 1.
		for i in range(self.state_history):

			# Wrap around if the buffer is filled
			if self.crt_size == self.max_size:
				j = (ind - i) % self.max_size
				k = (ind - i + 1) % self.max_size
			else:
				j = ind - i
				k = (ind - i + 1).clip(min=0)
				# If j == -1, then we set state_not_done to 0.
				state_not_done *= (j + 1).clip(min=0, max=1).reshape(-1, 1, 1) #np.where(j < 0, state_not_done * 0, state_not_done)
				j = j.clip(min=0)

			# State should be all 0s if the episode terminated previously
			state[:, i] = self.state[j] * state_not_done
			next_state[:, i] = self.state[k] * next_not_done

			# If this was the first timestep, make everything previous = 0
			next_not_done *= state_not_done
			state_not_done *= (1. - self.first_timestep[j]).reshape(-1, 1, 1)

		return (
			torch.ByteTensor(state).to(self.device).float(),
			torch.LongTensor(self.action[ind]).to(self.device),
			torch.ByteTensor(next_state).to(self.device).float(),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def save(self, save_folder, chunk=int(1e5)):
		np.save(f"{save_folder}_action.npy", self.action[:self.crt_size])
		np.save(f"{save_folder}_reward.npy", self.reward[:self.crt_size])
		np.save(f"{save_folder}_not_done.npy", self.not_done[:self.crt_size])
		np.save(f"{save_folder}_first_timestep.npy", self.first_timestep[:self.crt_size])
		np.save(f"{save_folder}_replay_info.npy", [self.ptr, chunk])

		crt = 0
		end = min(chunk, self.crt_size + 1)
		while crt < self.crt_size + 1:
			np.save(f"{save_folder}_state_{end}.npy", self.state[crt:end])
			crt = end
			end = min(end + chunk, self.crt_size + 1)


	def load(self, save_folder, size=-1):
		reward_buffer = np.load(f"{save_folder}_reward.npy")
		size = min(int(size), self.max_size) if size > 0 else self.max_size
		self.crt_size = min(reward_buffer.shape[0], size)
		
		# Adjust crt_size if we're using a custom size
		size = min(int(size), self.max_size) if size > 0 else self.max_size
		self.crt_size = min(reward_buffer.shape[0], size)

		self.action[:self.crt_size] = np.load(f"{save_folder}_action.npy")[:self.crt_size]
		self.reward[:self.crt_size] = reward_buffer[:self.crt_size]
		self.not_done[:self.crt_size] = np.load(f"{save_folder}_not_done.npy")[:self.crt_size]
		self.first_timestep[:self.crt_size] = np.load(f"{save_folder}_first_timestep.npy")[:self.crt_size]

		self.ptr, chunk = np.load(f"{save_folder}_replay_info.npy")

		crt = 0
		end = min(chunk, self.crt_size + 1)
		while crt < self.crt_size + 1:
			self.state[crt:end] = np.load(f"{save_folder}_state_{end}.npy")
			crt = end
			end = min(end + chunk, self.crt_size + 1)


# Generic replay buffer for standard gym tasks
class StandardBuffer(object):
	def __init__(self, state_dim, batch_size, buffer_size, device):
		self.batch_size = batch_size
		self.max_size = int(buffer_size)
		self.device = device

		self.ptr = 0
		self.crt_size = 0

		self.state = np.zeros((self.max_size, state_dim))
		self.action = np.zeros((self.max_size, 1))
		self.next_state = np.array(self.state)
		self.reward = np.zeros((self.max_size, 1))
		self.not_done = np.zeros((self.max_size, 1))


	def add(self, state, action, next_state, reward, done, episode_done, episode_start):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.crt_size = min(self.crt_size + 1, self.max_size)


	def sample(self):
		ind = np.random.randint(0, self.crt_size, size=self.batch_size)
		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.LongTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def save(self, save_folder):
		np.save(f"{save_folder}_state.npy", self.state[:self.crt_size])
		np.save(f"{save_folder}_action.npy", self.action[:self.crt_size])
		np.save(f"{save_folder}_next_state.npy", self.next_state[:self.crt_size])
		np.save(f"{save_folder}_reward.npy", self.reward[:self.crt_size])
		np.save(f"{save_folder}_not_done.npy", self.not_done[:self.crt_size])
		np.save(f"{save_folder}_ptr.npy", self.ptr)


	def load(self, save_folder, size=-1):
		reward_buffer = np.load(f"{save_folder}_reward.npy")
		
		# Adjust crt_size if we're using a custom size
		size = min(int(size), self.max_size) if size > 0 else self.max_size
		self.crt_size = min(reward_buffer.shape[0], size)

		self.state[:self.crt_size] = np.load(f"{save_folder}_state.npy")[:self.crt_size]
		self.action[:self.crt_size] = np.load(f"{save_folder}_action.npy")[:self.crt_size]
		self.next_state[:self.crt_size] = np.load(f"{save_folder}_next_state.npy")[:self.crt_size]
		self.reward[:self.crt_size] = reward_buffer[:self.crt_size]
		self.not_done[:self.crt_size] = np.load(f"{save_folder}_not_done.npy")[:self.crt_size]

		print(f"Replay Buffer loaded with {self.crt_size} elements.")


# Atari Preprocessing
# Code is based on https://github.com/openai/gym/blob/master/gym/wrappers/atari_preprocessing.py
class AtariPreprocessing(object):
	def __init__(
		self,
		env,
		frame_skip=4,
		frame_size=84,
		state_history=4,
		done_on_life_loss=False,
		reward_clipping=True, # Clips to a range of -1,1
		max_episode_timesteps=27000
	):
		self.env = env.env
		self.done_on_life_loss = done_on_life_loss
		self.frame_skip = frame_skip
		self.frame_size = frame_size
		self.reward_clipping = reward_clipping
		self._max_episode_steps = max_episode_timesteps
		self.observation_space = np.zeros((frame_size, frame_size))
		self.action_space = self.env.action_space

		self.lives = 0
		self.episode_length = 0

		# Tracks previous 2 frames
		self.frame_buffer = np.zeros(
			(2,
			self.env.observation_space.shape[0],
			self.env.observation_space.shape[1]),
			dtype=np.uint8
		)
		# Tracks previous 4 states
		self.state_buffer = np.zeros((state_history, frame_size, frame_size), dtype=np.uint8)


	def reset(self):
		self.env.reset()
		self.lives = self.env.ale.lives()
		self.episode_length = 0
		self.env.ale.getScreenGrayscale(self.frame_buffer[0])
		self.frame_buffer[1] = 0

		self.state_buffer[0] = self.adjust_frame()
		self.state_buffer[1:] = 0
		return self.state_buffer


	# Takes single action is repeated for frame_skip frames (usually 4)
	# Reward is accumulated over those frames
	def step(self, action):
		total_reward = 0.
		self.episode_length += 1

		for frame in range(self.frame_skip):
			_, reward, done, _ = self.env.step(action)
			total_reward += reward

			if self.done_on_life_loss:
				crt_lives = self.env.ale.lives()
				done = True if crt_lives < self.lives else done
				self.lives = crt_lives

			if done: 
				break

			# Second last and last frame
			f = frame + 2 - self.frame_skip 
			if f >= 0:
				self.env.ale.getScreenGrayscale(self.frame_buffer[f])

		self.state_buffer[1:] = self.state_buffer[:-1]
		self.state_buffer[0] = self.adjust_frame()

		done_float = float(done)
		if self.episode_length >= self._max_episode_steps:
			done = True

		return self.state_buffer, total_reward, done, [np.clip(total_reward, -1, 1), done_float]


	def adjust_frame(self):
		# Take maximum over last two frames
		np.maximum(
			self.frame_buffer[0],
			self.frame_buffer[1],
			out=self.frame_buffer[0]
		)

		# Resize
		image = cv2.resize(
			self.frame_buffer[0],
			(self.frame_size, self.frame_size),
			interpolation=cv2.INTER_AREA
		)
		return np.array(image, dtype=np.uint8)


	def seed(self, seed):
		self.env.seed(seed)


# Create environment, add wrapper if necessary and create env_properties
# modified to allow discrete state space as well
def make_env(env_name, atari_preprocessing):
	if env_name == 'Riverswim':
		env = riverSwim(20)
		is_atari = False
	else:
		env = gym.make(env_name)
		is_atari = gym.envs.registry.spec(env_name).entry_point == 'gym.envs.atari:AtariEnv'
		
	env = AtariPreprocessing(env, **atari_preprocessing) if is_atari else env

	if is_atari:
		state_dim = (
		atari_preprocessing["state_history"], 
		atari_preprocessing["frame_size"], 
		atari_preprocessing["frame_size"]
	)  
	elif env_name == 'Riverswim':
		state_dim = env.observation_dim
	else:# one of the toy examples (i.e. Taxi)
		state_dim = 1#env.observation_space.shape[0]

	return (
		env,
		is_atari,
		state_dim,
		env.action_space.n
	)

###########################################################################
################  openAI wrapper for Riverswim environment ################
###########################################################################


import gym
from gym import spaces

class riverSwim(gym.Env):
    """Riverswim Environment that follows gym interface"""
    #metadata = {'render.modes': ['human']}
    def __init__(self,episode_length):
        super(riverSwim, self).__init__()
        self.time = 0
        self.state = 0
        self.episode_length = episode_length
        self.done = False
        self._max_episode_steps = episode_length -1
        # Define action and observation space
        N_DISCRETE_ACTIONS = 2
        N_DISCRETE_STATES = 6 
        N_TIME_STEPS = episode_length
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.MultiDiscrete((N_DISCRETE_STATES, N_TIME_STEPS))
        self.observation_dim = 2
    # Execute one time step within the environment
    def step(self,action):
        if action not in [0,1]:
            return 'error'        
        if self.time == self.episode_length:
            self.done = True
        self.time += 1
        self.reward = 0 # no reward
        if self.state == 0:
            if action == 1: # swim to the right
                if np.random.binomial(1,.6)==1: # w.p. = 0.6 get to the right otherwise stay in state = 0
                    self.state = 1                
            else: #action == 0, stays in state = 0 and has the reward 5/1000
                self.reward = 5/1000
        elif self.state == 5:
            if action == 1: # swim to the right
                if np.random.binomial(1,.6)==1: # w.p. = 0.6 swim succesfully to the right 
                    self.reward = 1
                else: # w.p. 0.4 current takes it to the left
                    self.state = 4
            else: # action == 0
                self.state = 4
        else: #states 1,2,3,4
            if action == 1: # swim to the right
                dice = np.random.choice(3, 1, p=[.05,0.6,0.35])
                if dice==0: # w.p. = 0.05 current takes it to the left
                    self.state -= 1
                elif dice==2: # w.p. = 0.6 current it stays in the same state, w.p. 0.35 gets to the right
                    self.state += 1
            else: # action == 0
                self.state -= 1
        info = 'na'
        return np.array([self.state, self.time]),self.reward,self.done,info
    def reset(self):
    # Reset the state of the environment to an initial state
        self.time = 0
        self.state = 0
        self.done = False
        return np.array([self.state, self.time])
