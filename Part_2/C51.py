from turtle import update
import gymnasium as gym
import numpy as np
import utils.envs, utils.seed, utils.buffers, utils.torch
import torch
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# C51
# Based on Slide 11
# cs.uwaterloo.ca/~ppoupart/teaching/cs885-winter22/slides/cs885-module5.pdf

# Constants
SEEDS = [1, 2, 3, 4, 5]
t = utils.torch.TorchHelper()
DEVICE = t.device
OBS_N = 4               # State space size
ACT_N = 2               # Action space size
STARTING_EPSILON = 1.0  # Starting epsilon
STEPS_MAX = 10000       # Gradually reduce epsilon over these many steps
EPSILON_END = 0.1       # At the end, keep epsilon at this value
MINIBATCH_SIZE = 64     # How many examples to sample per train step
GAMMA = 0.99            # Discount factor in episodic reward objective
LEARNING_RATE = 5e-4    # Learning rate for Adam optimizer
TRAIN_AFTER_EPISODES = 10   # Just collect episodes for these many episodes
TRAIN_EPOCHS = 25        # Train for these many epochs every time
BUFSIZE = 10000         # Replay buffer size
EPISODES = 500          # Total number of episodes to learn over
TEST_EPISODES = 10      # Test episodes
HIDDEN = 512            # Hidden nodes
TARGET_NETWORK_UPDATE_FREQ = 10 # Target network update frequency

# C51 specific constants
ATOMS = 51              # Number of atoms for distributional network
VMIN = 0                # Minimum value for support
VMAX = 500              # Maximum value for support (updated for longer episodes)

# Global variables
EPSILON = STARTING_EPSILON
Z = None

# Create support for distributional RL
# This is the set of possible return values
support = torch.linspace(VMIN, VMAX, ATOMS).to(DEVICE)
delta_z = (VMAX - VMIN) / (ATOMS - 1)

# Create environment
# Create replay buffer
# Create distributional networks
# Create optimizer
def create_everything(seed):
    utils.seed.seed(seed)
    env = utils.envs.TimeLimit(utils.envs.NoisyCartPole(), 500)
    test_env = utils.envs.TimeLimit(utils.envs.NoisyCartPole(), 500)

    env.reset(seed=seed)
    test_env.reset(seed=seed)
    buf = utils.buffers.ReplayBuffer(BUFSIZE)
    Z = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N*ATOMS)
    ).to(DEVICE)
    Zt = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N*ATOMS)
    ).to(DEVICE)
    Zt.load_state_dict(Z.state_dict())
    OPT = torch.optim.Adam(Z.parameters(), lr = LEARNING_RATE)
    return env, test_env, buf, Z, Zt, OPT

# Create epsilon-greedy policy
def policy(env, obs):

    global EPSILON, EPSILON_END, STEPS_MAX, Z
    obs = t.f(obs).view(-1, OBS_N)  # Convert to torch tensor
    
    # With probability EPSILON, choose a random action
    # Rest of the time, choose argmax_a Q(s, a) 
    if np.random.rand() < EPSILON:
        action = np.random.randint(ACT_N)
    else:
        # Get distribution for each action
        with torch.no_grad():
            # Z network outputs logits for each (action, atom) pair
            logits = Z(obs).view(-1, ACT_N, ATOMS)  # (batch, actions, atoms)
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=2)  # (batch, actions, atoms)
            
            # Compute Q value for each action as expectation: Q(s,a) = Σ p_i * z_i
            q_values = (probs * support.view(1, 1, -1)).sum(dim=2)  # (batch, actions)
            
            # Select greedy action
            action = torch.argmax(q_values, dim=1).item()
    
    # Epsilon update rule: Keep reducing a small amount over
    # STEPS_MAX number of steps, and at the end, fix to EPSILON_END
    EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))
    
    return action

# Update networks
def update_networks(epi, buf, Z, Zt, OPT):
    # 從 replay buffer 取樣 (s, a, r, s', done)
    S, A, R, S2, D = buf.sample(MINIBATCH_SIZE, t)
    batch_size = S.size(0)

    # ----- 當前網路：得到被執行動作的 log prob 分佈 -----
    # Z(S) shape: (B, ACT_N * ATOMS) -> reshape 成 (B, ACT_N, ATOMS)
    logits = Z(S).view(batch_size, ACT_N, ATOMS)
    # 先對所有 (s,a) 做 softmax，再取 log
    log_probs_all = F.log_softmax(logits, dim=2)   # (B, ACT_N, ATOMS)

    # 用 gather 取出每個樣本實際執行的動作對應的 log prob
    # A: (B,) -> (B,1,1) 再擴展到 (B,1,ATOMS)
    A_idx = A.long().view(-1, 1, 1).expand(-1, 1, ATOMS)
    log_probs = log_probs_all.gather(1, A_idx).squeeze(1)  # (B, ATOMS)

    # ----- 目標網路：計算投影後的 target 分佈 m -----
    with torch.no_grad():
        # Zt(S2) -> (B, ACT_N, ATOMS)
        next_logits = Zt(S2).view(batch_size, ACT_N, ATOMS)
        next_probs = F.softmax(next_logits, dim=2)  # (B, ACT_N, ATOMS)

        # 計算下一狀態每個動作的 Q 值：E[Z] = Σ p_i z_i
        q_next = (next_probs * support.view(1, 1, -1)).sum(dim=2)  # (B, ACT_N)

        # 選擇 argmax_a' Q(s', a')
        best_actions = q_next.argmax(dim=1)  # (B,)

        # 取出對應 best action 的機率分佈 p(s', a*)
        probs_next_a = next_probs[torch.arange(batch_size, device=DEVICE), best_actions]  # (B, ATOMS)

        # Bellman 傳播後的支撐點 Tz = r + γ z_j (若非終止)
        not_done = (1.0 - D.float()).unsqueeze(1)  # (B,1)
        Tz = R.unsqueeze(1) + not_done * GAMMA * support.view(1, -1)  # (B, ATOMS)

        # 將 Tz 限制在 [VMIN, VMAX]
        Tz = Tz.clamp(VMIN, VMAX)

        # 計算投影位置 b_j = (Tz_j - VMIN) / Δz
        b = (Tz - VMIN) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        # 保險起見再 clamp 一次邊界
        l = l.clamp(0, ATOMS - 1)
        u = u.clamp(0, ATOMS - 1)

        # 建立 target 分佈 m
        m = torch.zeros(batch_size, ATOMS, device=DEVICE)

        # 展平 index 方便 scatter_add_
        offset = (torch.arange(batch_size, device=DEVICE) * ATOMS).unsqueeze(1)  # (B,1)

        # 低、高索引對應的權重
        w_l = probs_next_a * (u.float() - b)
        w_u = probs_next_a * (b - l.float())

        # 將機率質量 scatter 到 m 上
        m.view(-1).scatter_add_(0, (l + offset).view(-1), w_l.view(-1))
        m.view(-1).scatter_add_(0, (u + offset).view(-1), w_u.view(-1))

    # ----- Cross-Entropy / KL 損失 -----
    loss = -(m * log_probs).sum(dim=1).mean()

    # 反向傳播與梯度裁剪
    OPT.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(Z.parameters(), 10.0)
    OPT.step()

    # 定期同步 target network
    if epi % TARGET_NETWORK_UPDATE_FREQ == 0:
        Zt.load_state_dict(Z.state_dict())

    return loss.item()


# Play episodes
# Training function
def train(seed):

    global EPSILON, Z
    print("Seed=%d" % seed)

    # Create environment, buffer, Z, Z target, optimizer
    env, test_env, buf, Z, Zt, OPT = create_everything(seed)

    # epsilon greedy exploration
    EPSILON = STARTING_EPSILON

    testRs = [] 
    last25testRs = []
    print("Training:")
    pbar = tqdm.trange(EPISODES)
    for epi in pbar:

        # Play an episode and log episodic reward
        S, A, R = utils.envs.play_episode_rb(env, policy, buf)
        
        # Train after collecting sufficient experience
        if epi >= TRAIN_AFTER_EPISODES:

            # Train for TRAIN_EPOCHS
            for tri in range(TRAIN_EPOCHS): 
                update_networks(epi, buf, Z, Zt, OPT)

        # Evaluate for TEST_EPISODES number of episodes
        Rews = []
        for epj in range(TEST_EPISODES):
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
# Clip the curves to be between 0, 500
def plot_arrays(vars, color, label):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    plt.plot(range(len(mean)), mean, color=color, label=label)
    plt.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std,500), color=color, alpha=0.3)

if __name__ == "__main__":

    # Train for different seeds
    curves = []
    for seed in SEEDS:
        curves += [train(seed)]

    # Plot the curve for the given seeds
    plot_arrays(curves, 'b', 'c51')
    plt.legend(loc='best')
    plt.show()