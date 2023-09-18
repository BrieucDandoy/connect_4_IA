import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(self.state_size[0] * self.state_size[1], 64)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.action_size)
        self.softmax = nn.Softmax()

    def forward(self, state):
        x = state.view(state.size(0), -1)  # Flattening the input state
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        q_values = self.fc3(x)
        probabilities = self.softmax(q_values)
        return probabilities
