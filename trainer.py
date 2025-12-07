import gymnasium as gym
import torch
from torch.distributions import Categorical
from ppo_agent import PPO, device


def train_session(config):
    """
    학습 세션을 실행하고, UI 업데이트를 위해 중간 결과를 yield하는 제너레이터
    """
    env = gym.make('LunarLander-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Config 객체를 Agent에 전달
    model = PPO(state_dim, action_dim, config).to(device)

    score = 0.0
    print_interval = 20

    for n_epi in range(config.max_episodes):
        s, _ = env.reset()
        done = False
        truncated = False

        while not done and not truncated:
            for t in range(config.T_horizon):
                prob = model.pi(torch.from_numpy(s).float().to(device))
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)

                model.put_data((s, a, r / 100.0, s_prime, prob[a].item(), done))
                s = s_prime
                score += r

                if done or truncated:
                    break

            model.train_net()

        # UI 업데이트를 위한 데이터 반환
        if n_epi % print_interval == 0 and n_epi != 0:
            avg_score = score / print_interval
            score = 0.0

            # yield를 통해 app.py로 데이터 전송 (UI 갱신용)
            yield n_epi, avg_score

    env.close()