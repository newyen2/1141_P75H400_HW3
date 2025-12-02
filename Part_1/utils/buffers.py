import collections
import numpy as np
import random
import torch

# Replay buffer
class ReplayBuffer:
    
    # create replay buffer of size N
    def __init__(self, N, recurrent=False):
        self.buf = collections.deque(maxlen=N)
        self.recurrent = recurrent
        self.current_episode = []
    
    # add: add a transition (s, a, r, s2, d)
    def add(self, s, a, r, s2, d):
        if self.recurrent:
            # Store transitions as part of current episode
            self.current_episode.append((s, a, r, s2, d))
            
            # If episode is done, store the entire episode
            if d:
                self.buf.append(self.current_episode)
                self.current_episode = []
        else:
            self.buf.append((s, a, r, s2, d))
    
    # sample: return minibatch of size n
    def sample(self, n, t):
        if self.recurrent:
            # Sample n episodes
            n = min(n, len(self.buf))
            episodes = random.sample(self.buf, n)
            
            # Return list of episodes, each episode is (S, A, R, S2, D)
            result = []
            for episode in episodes:
                S, A, R, S2, D = [], [], [], [], []
                for transition in episode:
                    s, a, r, s2, d = transition
                    S.append(s)
                    A.append(a)
                    R.append(r)
                    S2.append(s2)
                    D.append(d)
                result.append((S, A, R, S2, D))
            
            return result
        else:
            # Original sampling for non-recurrent
            minibatch = random.sample(self.buf, n)
            S, A, R, S2, D = [], [], [], [], []
            
            for mb in minibatch:
                s, a, r, s2, d = mb
                S += [s]; A += [a]; R += [r]; S2 += [s2]; D += [d]

            if type(A[0]) == int:
                return t.f(S), t.l(A), t.f(R), t.f(S2), t.i(D)
            elif type(A[0]) == float:
                return t.f(S), t.f(A), t.f(R), t.f(S2), t.i(D)
            else:
                return t.f(S), torch.stack(A), t.f(R), t.f(S2), t.i(D)

