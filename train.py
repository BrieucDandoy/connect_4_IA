from game import connect4
from net import DQN
import random
import torch
import torch.optim as optim

state_size = (6, 7)  
action_size = 7 
learning_rate = 0.001
batch_size = 32
gamma = 0.995
epsilon = 1.0
epsilon_decay = 0.9
epsilon_min = 0.01
target_update_frequency = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


agent1_policy_net = DQN(state_size, action_size).to(device)
agent1_target_net = DQN(state_size, action_size).to(device)
agent1_target_net.load_state_dict(agent1_policy_net.state_dict())
agent1_target_net.eval()

agent2_policy_net = DQN(state_size, action_size).to(device)
agent2_target_net = DQN(state_size, action_size).to(device)
agent2_target_net.load_state_dict(agent2_policy_net.state_dict())
agent2_target_net.eval()

agent1_optimizer = optim.Adam(agent1_policy_net.parameters(), lr=learning_rate)
agent2_optimizer = optim.Adam(agent2_policy_net.parameters(), lr=learning_rate)

def select_action(state, agent_policy_net, epsilon):
    if random.random() < epsilon:
        return random.randint(0, action_size - 1)
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = agent_policy_net(state)
            action = q_values.argmax()
            
            return action

def update_network(agent_policy_net, agent_target_net, agent_optimizer, state, action, next_state, reward, done, gamma):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
    action = torch.LongTensor([action]).to(device)
    reward = torch.FloatTensor([reward]).to(device)
    done = torch.FloatTensor([done]).to(device)

    q_values = agent_policy_net(state).gather(1, action.unsqueeze(1))
    next_q_values = agent_target_net(next_state).max(1)[0].detach()
    expected_q_values = reward + gamma * next_q_values * (1 - done)

    loss = torch.nn.functional.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))
    agent_optimizer.zero_grad()
    loss.backward()
    agent_optimizer.step()

num_episodes = 10000
for episode in range(num_episodes):
    jeu = connect4()
    jeu.get_random_position()
    state = jeu.grille
    total_reward_agent1 = 0
    total_reward_agent2 = 0

    done = False
    while not done:
        action_agent1 = select_action(state, agent1_policy_net, epsilon)
        next_state, reward_agent1, done= jeu.play_NN(action_agent1,1)
        total_reward_agent1 += reward_agent1

        update_network(agent1_policy_net, agent1_target_net, agent1_optimizer, state, action_agent1, next_state, reward_agent1, done, gamma)

        # Si la partie est terminée après le coup de l'agent 1, on n'exécute pas l'agent 2
        if done:
            break

        action_agent2 = select_action(state, agent2_policy_net, epsilon)
        next_state, reward_agent2, done = jeu.play_NN(action_agent2,0.5)
        total_reward_agent2 += reward_agent2

        update_network(agent2_policy_net, agent2_target_net, agent2_optimizer, state, action_agent2, next_state, reward_agent2, done, gamma)

        # Passage à l'état suivant
        state = next_state

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Copie des poids de l'agent policy vers l'agent target
    if episode % target_update_frequency == 0:
        agent1_target_net.load_state_dict(agent1_policy_net.state_dict())
        agent2_target_net.load_state_dict(agent2_policy_net.state_dict())

    print(f"Episode {episode+1}: Agent 1 - Reward: {total_reward_agent1}, Agent 2 - Reward: {total_reward_agent2}")


torch.save(agent1_target_net.state_dict(), 'model/agent1')
torch.save(agent2_target_net.state_dict(), 'model/agent2')
