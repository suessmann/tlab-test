import torch
import torch.nn as nn

import pyrallis
import wandb

import os
import numpy as np
import random

from copy import deepcopy

from buffer import ReplayBuffer

# from buffer import TensorBatch

from dataclasses import dataclass, asdict

import gym
import d4rl

from tqdm import trange
import uuid

from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class TrainConfig:
    project: str = "CORL"
    group: str = "AWAC-D4RL"
    name: str = "AWAC"
    checkpoints_path: Optional[str] = "./checkpoints/"

    env_name: str = "halfcheetah-medium-v2"
    seed: int = 42
    test_seed: int = 69
    deterministic_torch: bool = False
    device: str = "cpu"

    buffer_size: int = 1_000_000
    num_train_ops_offline: int = 1_000_000
    batch_size: int = 256
    eval_frequency: int = 1000
    n_test_episodes: int = 10
    normalize_reward: bool = False

    num_train_ops_online: int = 500_000
    num_envs: int = 8
    add_from_dataset: int = 0
    updates_before_collections = 1_000

    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 5e-3
    lambda_awac: float = 1.0

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        device: str,
        min_action=-1.0,
        max_action=1.0,
        min_log_std=-20.0,
        max_log_std=2.0,
    ):
        super(Actor, self).__init__()
        self.device = device
        self.min_action, self.max_action = min_action, max_action
        self.min_log_std, self.max_log_std = min_log_std, max_log_std

        self.embed = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

    def forward(self, state):
        embed = self.embed(state)
        mu = self.mu(embed)
        log_std_ = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)

        dist = torch.distributions.Normal(mu, log_std_.exp())
        action = dist.rsample()
        action.clamp_(self.min_action, self.max_action)

        return action, dist

    @torch.no_grad()
    def get_action(self, state, evaluation=False) -> np.ndarray:
        if state.ndim != 2:
            state = state[None, ...]

        state = torch.tensor(state, dtype=torch.float32, device=self.device)

        _, prob = self.forward(state)
        action = prob.mean if evaluation else prob.sample()
        action.clamp_(self.min_action, self.max_action)

        if evaluation:
            action = action[0]

        action = action.cpu().numpy()

        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()

        self.q_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        q_val = self.q_net(state_action)

        return q_val


class AWAC:
    def __init__(
        self,
        actor: nn.Module,
        critic_1: nn.Module,
        critic_2: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer_1: torch.optim.Optimizer,
        critic_optimizer_2: torch.optim.Optimizer,
        gamma: float,
        tau: float,
        lambda_: float,
    ):
        self.actor = actor
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.target_critic_1 = deepcopy(critic_1)
        self.target_critic_2 = deepcopy(critic_2)

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer_1 = critic_optimizer_1
        self.critic_optimizer_2 = critic_optimizer_2
        self.gamma = gamma
        self.tau = tau
        self.lambda_inv = 1 / lambda_

    def actor_loss(self, state, action):
        action_pi, dist = self.actor(state)
        with torch.no_grad():
            v = torch.min(
                self.critic_1(state, action_pi),
                self.critic_2(state, action_pi),
            )
            q = torch.min(
                self.critic_1(state, action),
                self.critic_2(state, action),
            )
            assert v.shape == q.shape
            adv = torch.exp((q - v) * self.lambda_inv)
            adv = torch.clamp_max(adv, 100)
            # adv = torch.softmax(adv, -1)

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        assert log_prob.shape == adv.shape

        actor_loss = -1 * (log_prob * adv.detach()).mean()  # paranoidal .detach()

        return actor_loss, torch.linalg.vector_norm(adv)

    def critic_loss(self, state, action, reward, next_state, done):
        with torch.no_grad():
            next_action = self.actor(next_state)[0]
            q_next_min = torch.min(
                self.target_critic_1(next_state, next_action),
                self.target_critic_2(next_state, next_action),
            )
            target = reward + self.gamma * (1 - done) * q_next_min
            assert target.shape == (state.shape[0], 1)

        q_1 = self.critic_1(state, action)
        q_2 = self.critic_2(state, action)

        assert q_1.shape == target.shape
        assert q_2.shape == target.shape

        critic_loss_1 = nn.MSELoss()(q_1, target)
        critic_loss_2 = nn.MSELoss()(q_2, target)
        critic_loss = critic_loss_1 + critic_loss_2

        return critic_loss

    def update(self, batch) -> Dict[str, float]:
        state, action, reward, next_state, done = batch

        # update critic networks
        critic_loss = self.critic_loss(state, action, reward, next_state, done)
        self.critic_optimizer_1.zero_grad()
        self.critic_optimizer_2.zero_grad()
        critic_loss.backward()
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.step()

        # update actor network
        actor_loss, adv = self.actor_loss(state, action)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target networks
        self._soft_update(self.target_critic_1, self.critic_1)
        self._soft_update(self.target_critic_2, self.critic_2)

        info = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "adv": adv.mean().item(),
        }

        return info

    def _soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic_1.load_state_dict(state_dict["critic_1"])
        self.critic_2.load_state_dict(state_dict["critic_2"])


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def create_controller(
    state_dim: int,
    action_dim: int,
    config: TrainConfig,
) -> AWAC:
    actor = Actor(state_dim, action_dim, config.hidden_dim, config.device).to(
        config.device
    )
    critic_1 = Critic(state_dim, action_dim, config.hidden_dim).to(config.device)
    critic_2 = Critic(state_dim, action_dim, config.hidden_dim).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_1_optimizer = torch.optim.Adam(
        critic_1.parameters(), lr=config.learning_rate
    )
    critic_2_optimizer = torch.optim.Adam(
        critic_2.parameters(), lr=config.learning_rate
    )

    controller = AWAC(
        actor=actor,
        critic_1=critic_1,
        critic_2=critic_2,
        actor_optimizer=actor_optimizer,
        critic_optimizer_1=critic_1_optimizer,
        critic_optimizer_2=critic_2_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        lambda_=config.lambda_awac,
    )

    return controller


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: Actor, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.get_action(state, evaluation=True)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    env = gym.wrappers.TransformObservation(env, normalize_state)
    return env


def evaluate(
    epoch: int,
    env_eval: gym.Env,
    controller: AWAC,
    config: TrainConfig,
    mode: str,
):
    modes = ["online", "offline"]
    if mode not in modes:
        raise ValueError("Mode should be either 'online' or 'offline'")

    # evaluate
    episode_rewards = eval_actor(
        env_eval,
        controller.actor,
        config.device,
        config.n_test_episodes,
        config.test_seed,
    )
    normalized_reward = env_eval.get_normalized_score(episode_rewards) * 100.0

    # log
    wandb.log(
        {
            "eval/reward": episode_rewards.mean(),
            "eval/normalized_reward": normalized_reward.mean(),
        },
        step=epoch,
    )
    wandb.log({f"{mode}/test_normalized_reward": normalized_reward.mean()}, step=epoch)

    # save
    torch.save(
        controller.state_dict(),
        os.path.join(config.checkpoints_path, f"{mode}", f"awac_{mode}_{epoch}.pt"),
    )


def collect_traj_from_env(
    envs: Union[gym.Env, gym.vector.AsyncVectorEnv], actor: Actor, buffer: ReplayBuffer
):
    obs, done = envs.reset(), np.zeros(len(envs.env_fns), dtype=bool)
    episode_reward = np.zeros(len(envs.env_fns))
    while not done.all():
        action = actor.get_action(obs)
        next_obs, reward, done, _ = envs.step(action)
        buffer.add_transitions(
            {
                "observations": obs,
                "actions": action,
                "rewards": reward,
                "next_observations": next_obs,
                "terminals": done,
            }
        )
        episode_reward += reward
        obs = next_obs

    return episode_reward


def train_online(
    env_eval: gym.Env,
    config: TrainConfig,
    dataset: Dict[str, np.ndarray] = None,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
):
    # env = gym.vector.make(f"{config.env_name}", num_envs=config.num_envs)
    envs = gym.vector.AsyncVectorEnv(
        [
            lambda: wrap_env(gym.make(config.env_name), state_mean, state_std)
            for _ in range(config.num_envs)
        ]
    )

    state_dim = env_eval.observation_space.shape[0]
    action_dim = env_eval.action_space.shape[0]

    buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)
    awac = create_controller(state_dim, action_dim, config)
    awac.load_state_dict(
        torch.load(
            os.path.join(
                config.checkpoints_path,
                "offline",
                f"awac_offline_{config.num_train_ops_offline}.pt",
            )
        )
    )

    if config.add_from_dataset > 0:
        assert dataset is not None, "Provide a dataset to add trajectories from"

        idx_to_add = np.random.randint(
            0, dataset["observations"].shape[0], config.add_from_dataset
        )
        buffer.add_transitions(
            {
                "observations": dataset["observations"][idx_to_add],
                "actions": dataset["actions"][idx_to_add],
                "rewards": dataset["rewards"][idx_to_add],
                "next_observations": dataset["next_observations"][idx_to_add],
                "dones": dataset["terminals"][idx_to_add],
            }
        )

    # populate buffer with existing policy trajectories
    _ = collect_traj_from_env(envs, awac.actor, buffer)

    e_global = config.num_train_ops_offline
    for e in trange(config.num_train_ops_online):
        if (e + 1) % config.updates_before_collections == 0:
            episode_reward = collect_traj_from_env(envs, awac.actor, buffer)
            wandb.log(
                {"online/train_episode_reward": episode_reward.mean()}, step=(e + 1)
            )

        batch = buffer.sample(config.batch_size)
        update_info = awac.update(batch)
        wandb.log(update_info, step=(e_global + e))

        if (e + 1) % config.eval_frequency == 0:
            evaluate(
                epoch=(e + 1),
                env_eval=env_eval,
                controller=awac,
                config=config,
                mode="online",
            )



def train_offline(
    env_eval: gym.Env, config: TrainConfig, dataset: Dict[str, np.ndarray]
):
    state_dim = env_eval.observation_space.shape[0]
    action_dim = env_eval.action_space.shape[0]

    buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)
    buffer.load_d4rl_dataset(dataset)

    awac = create_controller(state_dim, action_dim, config)

    print(f"Checkpoints path: {config.checkpoints_path}")
    os.makedirs(config.checkpoints_path, exist_ok=True)
    os.makedirs(os.path.join(config.checkpoints_path, "online"), exist_ok=True)
    os.makedirs(os.path.join(config.checkpoints_path, "offline"), exist_ok=True)
    with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
        pyrallis.dump(config, f)

    for e in trange(config.num_train_ops_offline):
        batch = buffer.sample(config.batch_size)
        update_info = awac.update(batch)
        wandb.log(update_info, step=e)

        if (e + 1) % config.eval_frequency == 0:
            evaluate(
                epoch=(e + 1),
                env_eval=env_eval,
                controller=awac,
                config=config,
                mode="offline",
            )


@pyrallis.wrap()
def main(config: TrainConfig):
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=config.name,
        # mode="disabled",  # DO NOT FORGET ME
    )

    env = gym.make(f"{config.env_name}")
    set_seed(config.seed, env, deterministic_torch=False)
    dataset = d4rl.qlearning_dataset(env)
    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-6)

    dataset["observations"] = (dataset["observations"] - state_mean) / state_std
    dataset["next_observations"] = (
        dataset["next_observations"] - state_mean
    ) / state_std

    env = wrap_env(gym.make(f"{config.env_name}"), state_mean, state_std)

    print("\nStarting **offline** training...")
    train_offline(env, config, dataset)
    print("\nStarting **online** training...")
    train_online(env, config, dataset)

    wandb.finish()


if __name__ == "__main__":
    main()
