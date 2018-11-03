import gym
import numpy as np
import time
import random
import math

env = gym.make('Pong-v0')
print(env.action_space)
#> Discrete(6)
print(env.observation_space)
#> Box(210,160,3)

# 236 236 236 ball
# 213 130  74 left
# 92 186  92 right

class Agent():
	def __init__(self):
		self._x_window_size = 12
		self._y_window_size = 4
		self._window_step = 2
		
		self._num_actions = 3
		self._explore_rate = 0.5
		self._learning_rate = 0.01
		self._discount = 0.9
		# naive version
		# self._Q = np.zeros(((self._x_window_size + 1)*(self._y_window_size + 1), self._num_actions), dtype=np.float)
		# self._Q += np.random.normal(0, 0.1, self._Q.shape)
		# naive version
		
		# smart version
		self._Q = np.zeros(((self._x_window_size + 1)*(self._y_window_size + 1), self._num_actions), dtype=np.float)
		for i in range(self._x_window_size + 1):
			for j in range(self._y_window_size + 1):
				for a in range(3):
					if a == 0:
						offset = 0
					if a == 1:
						offset = 1
					if a == 2 :
						offset = -1
						
					position = (i - self._x_window_size/2)/(self._x_window_size/2)
					self._Q[i*self._y_window_size + j][a] = 0.1 - (0.2 * (position + offset/self._window_step) ** 2)
				
		self._Q += np.random.normal(0, 0.01, self._Q.shape)		
		print(self._Q)
		# smart version
		self._i = 0
		self._random_action_decay_rate = 0.995
		self._random_action_prob = 0.05
		self._last_state = None
		self._last_action = None
		self.ball_direction = 2 #right
		self._last_b_y = None

	def _center(self,mask):
		assert mask.shape == (160,160)
		x = 0
		y = 0
		flag = False
		for i in range(160):
			if flag:
				break
			for j in range(160):
				if mask[i][j]:
					x = i
					y = j
					flag = True
					break
					
		flag = False			
		for i in range(0,-1):
			if flag:
				break
			for j in range(159,-1,-1):
				if mask[i][j]:
					x += i
					y += j
					flag = True
					break
		return x/2,y/2
	def _obs2vec(self,observation):
		canvas = observation[34:194,:,0]
		rightplayer_mask = (canvas == 92)
		leftplayer_mask = (canvas == 213)
		ball = (canvas == 236)
		
		r_x,r_y = self._center(rightplayer_mask)
		l_x,l_y = self._center(leftplayer_mask)
		b_x,b_y = self._center(ball)
		
		if b_x == 0 and b_y == 0:
			return None,None,None
		
		delta_x = (b_x - r_x)/self._window_step
		delta_y = (b_y - r_y - 4)/self._window_step
		
		if delta_x > self._x_window_size/2:
			delta_x = self._x_window_size/2
		elif delta_x < - self._x_window_size/2:
			delta_x = - self._x_window_size/2
		
		if delta_y > self._y_window_size/2:
			delta_y = self._y_window_size/2
		elif delta_y < - self._y_window_size/2:
			delta_y = - self._y_window_size/2
		
		return int(delta_x + self._x_window_size/2), int(delta_y + self._y_window_size/2), b_y
		
	def _xy2state(self,x,y):
		return x * self._y_window_size + y
	def _state2xy(self,state):
		return state/self._y_window_size, state%self._y_window_size
	def getAction(self,observation,learning = True):
		x,y,b_y = self._obs2vec(observation)
		if x==None and y==None:
			# ball is not in the canvas
			return 0
		state = self._xy2state(x,y)
		
		reward = 0
		if self._last_b_y != None:
			if self._last_b_y>b_y:
				if self.ball_direction == 2:
					# catched the ball
					print("catched the ball")
					reward = 0.3
				self.ball_direction = 1
			if self._last_b_y<b_y:
				self.ball_direction = 2
		self._last_b_y = b_y	
			
		if self._last_state != None and self._last_action != None:
			self._Q[self._last_state][self._last_action] = (1 - self._learning_rate)*self._Q[self._last_state][self._last_action] + self._learning_rate * self._discount * (reward + np.max(self._Q[state,:]))
		
		explore_rate = self._random_action_decay_rate**self._i*self._random_action_prob
		
		if random.random() < explore_rate:
			action = int(random.random()*3)
		else:
			action = np.argmax(self._Q[state])
		
		self._last_action = action
		self._last_state = state
		if action == 0:
			return 0
		elif action == 1:
			return 2
		elif action == 2:
			return 3
			
	def getReward(self,reward):
		self._Q[self._last_state][self._last_action] = self._Q[self._last_state][self._last_action] *(1 - self._learning_rate) + self._learning_rate * reward
		self._last_state = None
		self._last_action = None
		self._i += 1
		self.ball_direction = 2
		self._last_b_y = None
		
agent = Agent()
# Training
		
observation = env.reset()
# in the firth 20 frame, the opponent and ball are invisible
for _ in range(21):
	env.step(0)
t = 0
reward_sum = 0
rewards = []
while(t<1000):
	env.render()
	action = agent.getAction(observation)
	observation, reward, done, info = env.step(action)
	if reward != 0:
		agent.getReward(reward)
		reward_sum+=reward
		t+=1
		print(t,reward,reward_sum/t)
		rewards.append(reward)
	if done:
		observation = env.reset()
		for _ in range(21):
			env.step(0)
print(rewards)
input()	