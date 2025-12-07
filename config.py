from dataclasses import dataclass

@dataclass
class PPOConfig:
    """PPO 하이퍼파라미터를 담는 데이터 클래스"""
    lr: float = 0.0005
    gamma: float = 0.99
    lmbda: float = 0.95
    eps_clip: float = 0.2
    K_epoch: int = 3
    T_horizon: int = 2048
    entropy_coef: float = 0.01
    max_episodes: int = 1000


# 실험 프리셋 (Presets) 정의
EXPERIMENT_PRESETS = {
    "1. Baseline (Stable PPO)": {
        "lr": 0.0005,
        "eps_clip": 0.2,
        "entropy_coef": 0.01,
        "description": "[성공 케이스] 논문 권장 설정. 안정적인 우상향 그래프 예상."
    },
    "2. High Clip (Policy Collapse)": {
        "lr": 0.0005,
        "eps_clip": 0.8,
        "entropy_coef": 0.01,
        "description": "[실패 케이스] Clipping이 너무 커서 학습이 불안정하거나 붕괴됨을 증명."
    },
    "3. No Entropy (Local Minima)": {
        "lr": 0.0005,
        "eps_clip": 0.2,
        "entropy_coef": 0.0,
        "description": "[비교 케이스] 탐험 없이 성급하게 수렴하여 최고 점수에 도달 못할 가능성 확인."
    }
}