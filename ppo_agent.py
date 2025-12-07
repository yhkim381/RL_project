import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# GPU/CPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        """
        config: config.py에서 정의한 PPOConfig 객체
        """
        super(PPO, self).__init__()
        self.data = []

        # Config 객체에서 변수 로드
        self.lr = config.lr
        self.gamma = config.gamma
        self.lmbda = config.lmbda
        self.eps_clip = config.eps_clip
        self.K_epoch = config.K_epoch
        self.entropy_coef = config.entropy_coef

        # Neural Networks
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_pi = nn.Linear(128, action_dim)
        self.fc_v = nn.Linear(128, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s = torch.tensor(np.array(s_lst), dtype=torch.float).to(device)
        a = torch.tensor(np.array(a_lst)).to(device)
        r = torch.tensor(np.array(r_lst), dtype=torch.float).to(device)
        s_prime = torch.tensor(np.array(s_prime_lst), dtype=torch.float).to(device)
        done_mask = torch.tensor(np.array(done_lst), dtype=torch.float).to(device)
        prob_a = torch.tensor(np.array(prob_a_lst), dtype=torch.float).to(device)

        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(self.K_epoch):
            td_target = r + self.gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.cpu().detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(np.array(advantage_lst), dtype=torch.float).to(device)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

            dist = Categorical(pi)
            entropy = dist.entropy().mean()

            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach()) - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()