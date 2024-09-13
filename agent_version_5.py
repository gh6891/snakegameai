import torch
import random
import numpy as np
from collections import deque
from game2 import SnakeGameAI, Direction, Point
from model_version_3 import Atarimodel, QTrainer
from helper import plot
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MAX_MEMORY = 10000
BATCH_SIZE = 128
LR = 1e-4

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0.9 # randomness
        self.epsilon_min = 0.05
        self.epsilon_decay = 1000
        self.gamma = 0.99 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Atarimodel().to(device)
        self.target = Atarimodel().to(device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LR, amsgrad=True)
        self.trainer = QTrainer(self.model, self.target, lr=LR, gamma=self.gamma, device=device)
        self.frames = deque(maxlen = 4)
        self.step_count = 0
        self.TAU = 0.005

        for i in range(4):
            self.frames.append(np.zeros((84, 84)))


    def get_state(self, game):

        # 4 * 84 * 84로반환 해야함

        new_frame=game.get_downsampled_image() # 3*84*84
        # print(type(new_frame))
        new_frame=np.dot(new_frame[...,:3], [0.2989, 0.5870, 0.1140])#grayscale변환
        new_frame = np.transpose(new_frame)
        plt.imsave("new_frame.png", new_frame, cmap='gray')
        # print("new_frame.shape : ",new_frame.shape)
        self.frames.append(new_frame)
        state = np.stack(self.frames, axis= 0)
        # state = torch.tensor(state, dtype=torch.float32).to(device)
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    # def train_long_memory(self):
    #     if len(self.memory) > BATCH_SIZE:
    #         mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
    #     else:
    #         mini_sample = self.memory
        
    #     states, actions, rewards, next_states, dones = zip(*mini_sample)
    #     self.trainer.train_step(states, actions, rewards, next_states, dones, device)
    #     #for state, action, reward, nexrt_state, done in mini_sample:
    #     #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done, device)



    def get_action(self, state):

        final_move = [0,0,0]
        self.step_count += 1
        sample = random.random()
        eps_threshold = self.epsilon_min + (self.epsilon - self.epsilon_min) * math.exp(-1. * self.step_count / self.epsilon_decay)


        if sample < eps_threshold:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
            
            prediction = self.model(state0)
            
            move = torch.argmax(prediction).item()
        
        final_move[move] = 1

        return final_move


    def compute_loss(model, replay_buffer, batch_size, gamma, device=device):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(np.float32(state)).to(device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(device)
        action = torch.LongTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(done).to(device)

        q_values_old = model(state)
        q_values_new = model(next_state)

        q_value_old = q_values_old.gather(1, action.unsqueeze(1)).squeeze(1)
        q_value_new = q_values_new.max(1)[0]
        expected_q_value = reward + gamma * q_value_new * (1 - done)

        loss = (q_value_old - expected_q_value.data).pow(2).mean()

        return loss




    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        # print("1",type(states))  # 리스트인지 확인
        # print("2",len(states))  # 리스트 길이 확인
        # print("3",type(states[0]))  # 각 항목의 타입 확인
        # print("4",len(states[0]))  # 각 항목의 길이 확인

        states = np.array(states)
        next_states = np.array(next_states)

        # print("1",type(states))  # 리스트인지 확인
        # print("2",len(states))  # 리스트 길이 확인
        # print("3",type(states[0]))  # 각 항목의 타입 확인
        # print("4",len(states[0]))  # 각 항목의 길이 확인
        # print("5", states.shape)
        states = torch.tensor(states, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        actions_indices = actions.argmax(dim=1).unsqueeze(1)
        state_action_values = self.model(states).gather(1, actions_indices) # (128, 3)
        with torch.no_grad():
            next_state_values = self.target(next_states).max(1)[0] # 128
        next_state_values = next_state_values * (1 - dones)
        expected_state_action_values = (next_state_values * self.gamma) + rewards
        # with torch.no_grad():
        #     next_state_values[non_final_mask] = self.target(non_final_next_states).max(1).values
        # with torch.no_grad():
        #     next_state_values[]
        # expected_state_action_values = (next_state_values * self.gamma) + rewards

        criterion = nn.SmoothL1Loss()
        #이 아래에서 문제 발생
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            # 모델 최적화
        self.optimizer.zero_grad()
        loss.backward()
        # 변화도 클리핑 바꿔치기
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI(render_mode= True)
    
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        # if done:
        #     state_new = None
        # else:
        #     state_new = agent.get_state(game)

        # trainer 클래스 활용
        # agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            #현재있는 함수활용

            if score > record:
                record = score
                agent.model.save()

            print(f'Game: {agent.n_games}, Score: {score}, Record: {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
        agent.optimize_model()

        target_state_dict = agent.target.state_dict()
        model_state_dict = agent.model.state_dict()
        for key in model_state_dict:
            target_state_dict[key] = model_state_dict[key] * agent.TAU + target_state_dict[key]*(1-agent.TAU)
        agent.target.load_state_dict(target_state_dict)
# def play():
#     plot_scores = []
#     plot_mean_scores = []
#     total_score = 0
#     record = 0
#     agent = Agent()
#     game = SnakeGameAI()
#     while True:


if __name__ == '__main__':
    train()
    # fake = np.zeros((128,4,84,84)) # (1, 4, 84, 84)
    # print(fake.shape)
    # fake = torch.tensor(fake, dtype=torch.float32)
    # model = Atarimodel()
    # out = model(fake)
    # print(out)