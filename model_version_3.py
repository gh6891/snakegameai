import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
class Atarimodel(nn.Module):
    def __init__(self):
        super(Atarimodel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(in_features=32*9*9, out_features= 512)
        self.fc2 = nn.Linear(in_features=512, out_features = 3)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def save(self, file_name='model_new_version.pth'):
        model_folder_path = './model_version_3'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma, device):
        self.lr = lr
        self.gamma = gamma
        #model Linear_QNet((32, 24, 3), 256, 3)
        self.model = model.to(device)
        self.optimizer = optim.SGD(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done, device):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.long).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.float32).to(device)  
        # (n, x)
        if len(state.shape) == 3:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = torch.unsqueeze(done, 0)

        # print(state.shape)
        q_values = self.model(state)
        action_indices = action.argmax(dim=1)  # [1]
        q_values = q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)

        next_q_values = self.model(next_state).max(1)[0]

        expected_q_values = reward + self.gamma * next_q_values * (1 - done)

        # # 1: predicted Q values with current state
        # pred = self.model(state)

        # target = pred.clone()
        # for idx in range(len(done)):
        #     Q_new = reward[idx]
        #     if not done[idx]:
        #         next_state_input = next_state[idx].unsqueeze(0)
        #         Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state_input))

        #     target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(q_values, expected_q_values.detach())
        loss.backward()

        self.optimizer.step()

        # print(f"Loss: {loss.item()}")



