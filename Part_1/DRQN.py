import gymnasium as gym
import numpy as np
import random
import utils.envs, utils.seed, utils.buffers, utils.torch
import torch
import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Deep Recurrent Q Learning
# Slide 17
# cs.uwaterloo.ca/~ppoupart/teaching/cs885-winter22/slides/cs885-module4.pdf

# Constants
SEEDS = [1, 2, 3, 4, 5]
t = utils.torch.TorchHelper()
OBS_N = 2               # State space size
ACT_N = 2               # Action space size
STARTING_EPSILON = 1.0  # Starting epsilon
STEPS_MAX = 10000       # Gradually reduce epsilon over these many steps
EPSILON_END = 0.1       # At the end, keep epsilon at this value
MINIBATCH_SIZE = 64     # How many examples to sample per train step
GAMMA = 0.99            # Discount factor in episodic reward objective
LEARNING_RATE = 5e-4    # Learning rate for Adam optimizer
TRAIN_AFTER_EPISODES = 25   # Just collect episodes for these many episodes
TRAIN_EPOCHS = 10         # Train for these many epochs every time (reduced from 25)
BUFSIZE = 10000         # Replay buffer size
EPISODES = 2000         # Total number of episodes to learn over
TEST_EPISODES = 10      # Test episodes
HIDDEN = 512            # Hidden nodes
LSTM_HIDDEN = 256       # LSTM hidden size
TARGET_NETWORK_UPDATE_FREQ = 10 # Target network update frequency

# Global variables
EPSILON = STARTING_EPSILON
Q = None
hidden_state = None

# Deep recurrent Q network
class DRQN(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        # Input layer
        self.fc1 = torch.nn.Linear(OBS_N, HIDDEN)
        self.relu = torch.nn.ReLU()
        
        # LSTM layer
        self.lstm = torch.nn.LSTM(HIDDEN, LSTM_HIDDEN, batch_first=True)
        
        # Output layer
        self.fc2 = torch.nn.Linear(LSTM_HIDDEN, ACT_N)
    
    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len, obs_n) or (batch, obs_n)
        batch_size = x.size(0)
        
        # If input is 2D, add sequence dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, obs_n)
        
        # Pass through first fully connected layer
        x = self.fc1(x)  # (batch, seq_len, hidden)
        x = self.relu(x)
        
        # Pass through LSTM
        if hidden is None:
            x, new_hidden = self.lstm(x)
        else:
            x, new_hidden = self.lstm(x, hidden)
        
        # Take the last output from LSTM
        x = x[:, -1, :]  # (batch, lstm_hidden)
        
        # Pass through output layer
        q_values = self.fc2(x)  # (batch, act_n)
        
        return q_values, new_hidden
    
    def init_hidden(self, batch_size=1):
        # Initialize hidden state (h_0, c_0)
        h_0 = torch.zeros(1, batch_size, LSTM_HIDDEN)
        c_0 = torch.zeros(1, batch_size, LSTM_HIDDEN)
        return (h_0, c_0)


# Create environment
# Create replay buffer
# Create network for Q(s, a)
# Create target network
# Create optimizer
def create_everything(seed):
    utils.seed.seed(seed)
    env = utils.envs.TimeLimit(utils.envs.PartiallyObservableCartPole(), 200)
    test_env = utils.envs.TimeLimit(utils.envs.PartiallyObservableCartPole(), 200)
    env.reset(seed=seed)
    test_env.reset(seed=seed)
    buf = utils.buffers.ReplayBuffer(BUFSIZE, recurrent=True)
    Q = DRQN()
    Qt = DRQN()
    Qt.load_state_dict(Q.state_dict())
    OPT = torch.optim.Adam(Q.parameters(), lr = LEARNING_RATE)
    return env, test_env, buf, Q, Qt, OPT


# Create epsilon-greedy policy
# def policy(env, obs):
#     global EPSILON, EPSILON_END, STEPS_MAX, Q, hidden_state

#     obs = t.f(obs).view(1, -1)  # Convert to torch tensor (1, OBS_N)
    
#     # With probability EPSILON, choose a random action
#     # Rest of the time, choose argmax_a Q(s, a) 
#     if np.random.rand() < EPSILON:
#         action = np.random.randint(ACT_N)
#     else:
#         with torch.no_grad():
#             qvalues, hidden_state = Q(obs, hidden_state)
#             action = torch.argmax(qvalues).item()
    
#     # Epsilon update rule: Keep reducing a small amount over
#     # STEPS_MAX number of steps, and at the end, fix to EPSILON_END
#     EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))
    
#     return action

def policy(env, obs):
    global EPSILON, hidden_state, Q

    # 先把觀測轉成 batch=1, seq_len=1 的張量
    obs_tensor = t.f(obs).view(1, 1, -1)  # (1, 1, OBS_N)

    # 決定這一步要不要探索
    explore = (random.random() < EPSILON)

    if explore:
        # 純隨機選動作，不更新 hidden_state（跟原本一樣）
        action = random.randrange(ACT_N)
    else:
        # 利用模式：用目前的 hidden_state 經過 DRQN 得到 Q 值
        with torch.no_grad():
            q_values, hidden_state = Q(obs_tensor, hidden_state)
            # q_values: (1, ACT_N)，取出 scalar
            action = int(q_values.argmax(dim=1).item())

    # 退火：每一步都減少固定量，最低到 EPSILON_END
    decay_step = 1.0 / STEPS_MAX
    EPSILON -= decay_step
    if EPSILON < EPSILON_END:
        EPSILON = EPSILON_END

    return action

def _sequence_q_values(net, obs_seq):
    """
    obs_seq: (T, OBS_N) 的張量，一整集 (episode) 的觀測序列
    回傳: (T, ACT_N) 的 Q 值序列
    """
    # 加上 batch 維度 => (1, T, OBS_N)
    seq = obs_seq.unsqueeze(0)

    # 這裡每一集都用全 0 hidden state，等價於原本 update 裡每集呼叫 init_hidden
    h0, c0 = net.init_hidden(batch_size=1)

    # Linear + ReLU
    x = net.fc1(seq)
    x = net.relu(x)

    # LSTM
    lstm_out, _ = net.lstm(x, (h0, c0))   # (1, T, LSTM_HIDDEN)

    # 對每個 time step 做輸出層 => (T, ACT_N)
    q_seq = net.fc2(lstm_out.squeeze(0))
    return q_seq


# Update networks
# def update_networks(epi, buf, Q, Qt, OPT):
    
#     # Sample a minibatch of episodes
#     episodes = buf.sample(MINIBATCH_SIZE, t)
    
#     total_loss = 0.
#     batch_loss = 0.
    
#     # Process only a subset of episodes per update for efficiency
#     num_episodes_per_update = min(8, len(episodes))
#     sampled_episodes = random.sample(episodes, num_episodes_per_update)
    
#     for episode in sampled_episodes:
#         S, A, R, S2, D = episode
        
#         seq_len = len(S)
        
#         # Convert to tensors - more efficient
#         S_tensor = torch.stack([t.f(s) for s in S]).unsqueeze(0)  # (1, seq_len, obs_n)
#         A_tensor = t.l(A)  # (seq_len,)
#         R_tensor = t.f(R)  # (seq_len,)
#         S2_tensor = torch.stack([t.f(s) for s in S2]).unsqueeze(0)  # (1, seq_len, obs_n)
#         D_tensor = t.i(D)  # (seq_len,)
        
#         # Initialize hidden states
#         hidden = Q.init_hidden(batch_size=1)
        
#         # Forward pass through entire sequence at once for Q network
#         with torch.no_grad():
#             # For target network
#             target_hidden = Qt.init_hidden(batch_size=1)
#             # Process through LSTM
#             x = Qt.fc1(S2_tensor)
#             x = Qt.relu(x)
#             lstm_out, _ = Qt.lstm(x, target_hidden)
#             q2_values = Qt.fc2(lstm_out.squeeze(0))  # (seq_len, act_n)
#             max_q2_values = torch.max(q2_values, dim=1).values  # (seq_len,)
#             targets = R_tensor + GAMMA * max_q2_values * (1 - D_tensor.float())
        
#         # Forward pass for Q network
#         x = Q.fc1(S_tensor)
#         x = Q.relu(x)
#         lstm_out, _ = Q.lstm(x, hidden)
#         q_values = Q.fc2(lstm_out.squeeze(0))  # (seq_len, act_n)
        
#         # Gather Q values for taken actions
#         q_values_taken = q_values.gather(1, A_tensor.unsqueeze(1)).squeeze(1)  # (seq_len,)
        
#         # Compute loss
#         loss = torch.nn.MSELoss()(q_values_taken, targets.detach())
#         batch_loss += loss
#         total_loss += loss.item()
    
#     # Backpropagation on accumulated loss
#     OPT.zero_grad()
#     batch_loss.backward()
#     torch.nn.utils.clip_grad_norm_(Q.parameters(), 1.0)
#     OPT.step()

#     # Update target network
#     if epi % TARGET_NETWORK_UPDATE_FREQ == 0:
#         Qt.load_state_dict(Q.state_dict())

#     return total_loss / num_episodes_per_update

def update_networks(epi, buf, Q, Qt, OPT):
    # 從 replay buffer 抽出一批 episodes
    episodes = buf.sample(MINIBATCH_SIZE, t)
    if len(episodes) == 0:
        return 0.0

    # 跟原本一樣，只挑少量的 episodes 來更新，減少計算量
    num_episodes = min(8, len(episodes))
    indices = np.random.choice(len(episodes), size=num_episodes, replace=False)

    losses = []

    for idx in indices:
        S, A, R, S2, D = episodes[idx]

        # 轉成張量（這裡 shape 都是 (T, ·)）
        S_tensor  = torch.stack([t.f(s)  for s in S])   # (T, OBS_N)
        S2_tensor = torch.stack([t.f(s2) for s2 in S2]) # (T, OBS_N)
        A_tensor  = t.l(A)                              # (T,)
        R_tensor  = t.f(R)                              # (T,)
        D_tensor  = t.i(D).float()                      # (T,)

        # ----------------  計算 target: R_t + γ max_a' Q_tgt(s'_t, a')  ----------------
        with torch.no_grad():
            # 用 target 網路算下一狀態的 Q(s'_t, ·)
            q2_seq = _sequence_q_values(Qt, S2_tensor)     # (T, ACT_N)
            max_q2 = q2_seq.max(dim=1).values              # (T,)

            targets = R_tensor + GAMMA * max_q2 * (1.0 - D_tensor)

        # ----------------  計算 Q 網路對應動作的預測值 Q(s_t, a_t)  ----------------
        q_seq = _sequence_q_values(Q, S_tensor)            # (T, ACT_N)
        q_taken = q_seq.gather(1, A_tensor.view(-1, 1)).squeeze(1)  # (T,)

        # MSE loss（與原本相同）
        loss = torch.nn.functional.mse_loss(q_taken, targets)
        losses.append(loss)

    # 對這 batch 的 episodes 取平均 loss
    batch_loss = torch.stack(losses).mean()

    OPT.zero_grad()
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(Q.parameters(), 1.0)
    OPT.step()

    # 固定頻率同步 target network（跟原本一樣）
    if epi % TARGET_NETWORK_UPDATE_FREQ == 0:
        Qt.load_state_dict(Q.state_dict())

    return batch_loss.item()


# Play episodes
# Training function
def train(seed):

    global EPSILON, Q, hidden_state
    print("Seed=%d" % seed)

    # Create environment, buffer, Q, Q target, optimizer
    env, test_env, buf, Q, Qt, OPT = create_everything(seed)

    # epsilon greedy exploration
    EPSILON = STARTING_EPSILON

    testRs = [] 
    last25testRs = []
    print("Training:")
    pbar = tqdm.trange(EPISODES)
    for epi in pbar:

        # Reset hidden state at the beginning of each episode
        hidden_state = Q.init_hidden(batch_size=1)
        
        # Play an episode and log episodic reward
        S, A, R = utils.envs.play_episode_rb(env, policy, buf)
        
        # Train after collecting sufficient experience
        if epi >= TRAIN_AFTER_EPISODES:
            # Train for TRAIN_EPOCHS
            for tri in range(TRAIN_EPOCHS): 
                update_networks(epi, buf, Q, Qt, OPT)

        # Evaluate for TEST_EPISODES number of episodes
        Rews = []
        for epj in range(TEST_EPISODES):
            hidden_state = Q.init_hidden(batch_size=1)
            S, A, R = utils.envs.play_episode(test_env, policy, render = False)
            Rews += [sum(R)]
        testRs += [sum(Rews)/TEST_EPISODES]

        # Update progress bar
        last25testRs += [sum(testRs[-25:])/len(testRs[-25:])]
        pbar.set_description("R25(%g)" % (last25testRs[-1]))

    pbar.close()
    print("Training finished!")
    env.close()

    return last25testRs


# Plot mean curve and (mean-std, mean+std) curve with some transparency
# Clip the curves to be between 0, 200
def plot_arrays(vars, color, label):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    plt.plot(range(len(mean)), mean, color=color, label=label)
    plt.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std,200), color=color, alpha=0.3)


if __name__ == "__main__":

    # Train for different seeds
    curves = []
    for seed in SEEDS:
        curves += [train(seed)]

    # Plot the curve for the given seeds
    plot_arrays(curves, 'b', 'drqn')
    plt.legend(loc='best')
    plt.show()