import torch
import torch.nn as nn

import pyrallis
import wandb

import os
import numpy as np
import random

from copy import deepcopy

from buffer import ReplayBuffer

from dataclasses import dataclass, asdict

import gym
import d4rl

from tqdm import trange
import uuid

from typing import Any, Dict, List, Optional, Tuple, Union

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"


@dataclass
class TrainConfig:
    project: str = "tlab-test"
    group: str = "TD-BC-RF"
    name: str = "td-bc-rf"
    checkpoints_path: Optional[str] = "./checkpoints/"

    env_name: str = "halfcheetah-medium-v2"
    seed: int = 42
    test_seed: int = 69
    deterministic_torch: bool = False
    device: str = "cpu"

    buffer_size: int = 1_000_000
    batch_size: int = 256
    eval_frequency: int = 1000
    n_test_episodes: int = 10
    normalize_reward: bool = False
    num_envs: int = 8

    train_offline: bool = True
    train_refinement: bool = True
    train_online: bool = True

    num_train_ops_offline: int = 1_000_000
    num_train_ops_policy_refinement: int = 250_000
    num_train_ops_online: int = 500_000

    checkpoint_path_trained_offline: Optional[
        str
    ] = "/Users/Ilya.Zisman/uni/maga/online-rl/checkpoints/pretrained_tdbc/tdbcrf_offline_training_1000000.pt"
    add_from_dataset: int = 0
    updates_before_collections = 1_000

    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 5e-3

    lambda_td: float = 5.0
    alpha_start: float = 0.4
    alpha_finish: float = 0.2
    epsilon_noise_offline: float = 0.2
    epsilon_noise_online: float = 0.1

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


class TBR:  # TdBcRf
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
        lambda_td: float,
        alpha_start: float,
        alpha_finish: float,
        epsilon_noise_offline: float,
        epsilon_noise_online: float,
        online_budget: int,
    ):
        self.actor = actor
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.target_critic_1 = deepcopy(critic_1)
        self.target_critic_2 = deepcopy(critic_2)
        self.target_actor = deepcopy(actor)

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer_1 = critic_optimizer_1
        self.critic_optimizer_2 = critic_optimizer_2
        self.gamma = gamma
        self.tau = tau
        self.lambda_td = lambda_td
        self.alpha_prime = alpha_start / lambda_td
        self.alpha_start = alpha_start
        self.alpha_finish = alpha_finish
        self.alpha_online = alpha_start
        self.epsilon_noise_offline = epsilon_noise_offline
        self.epsilon_noise_online = epsilon_noise_online

        self.mode = None
        self.kappa = np.exp((1 / online_budget) * np.log(alpha_finish / alpha_start))

    def _soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )

    def select_mode(self, mode):
        if mode not in ["offline_training", "offline_refinement", "online_training"]:
            raise ValueError(
                "Training mode should either be 'offline_training', 'offline_refinement', 'online_training'"
            )

        self.mode = mode

        if mode == "online_training":
            self.actor.epsilon_noise = self.epsilon_noise_online
        if mode in ["offline_training", "offline_refinement"]:
            self.actor.epsilon_noise = self.epsilon_noise_offline

    def _calculate_online_alpha(self):
        assert self.mode == "online_training", "Controller should be in refinement mode"

        self.alpha_online = self.kappa * self.alpha_online

        return self.alpha_online

    def critic_loss(self, state, action, reward, next_state, done):
        with torch.no_grad():
            next_action = self.target_actor(next_state, add_noise=False)[0]
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

    def actor_loss(self, state, action):
        pi_action, _ = self.actor(state, add_noise=True)
        q_min = torch.min(
            self.critic_1(state, pi_action), self.critic_2(state, pi_action)
        )

        q_min = self._normalize_q(q_min)
        bc_loss = torch.nn.MSELoss()(action, pi_action)

        if self.mode == "offline_training":
            alpha = self.alpha_start
        elif self.mode == "offline_refinement":
            alpha = self.alpha_prime
        elif self.mode == "online_training":
            alpha = self._calculate_online_alpha()

        loss = (alpha * bc_loss - q_min).mean()
        info_carry = [q_min, bc_loss, alpha]

        return loss, info_carry

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
        actor_loss, info_carry = self.actor_loss(state, action)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        self.actor_optimizer.step()

        # update target networks
        self._soft_update(self.target_critic_1, self.critic_1)
        self._soft_update(self.target_critic_2, self.critic_2)
        self._soft_update(self.target_actor, self.actor)

        q_loss, bc_loss, alpha = info_carry
        info = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "q_loss": q_loss.mean().item(),
            "bc_loss": bc_loss.mean().item(),
            "alpha": alpha,
        }

        return info

    def _normalize_q(self, q):
        return q / q.abs().sum(-1).detach()

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
        epsilon_noise=0.2,
        noise_clip_min=-0.5,
        noise_clip_max=0.5,
    ):
        super(Actor, self).__init__()
        self.device = device
        self.min_action, self.max_action = min_action, max_action
        self.min_log_std, self.max_log_std = min_log_std, max_log_std
        self.noise_clip_min, self.noise_clip_max = noise_clip_min, noise_clip_max
        self.epsilon_noise = epsilon_noise

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

    def forward(self, state, add_noise=False):
        embed = self.embed(state)
        mu = self.mu(embed)
        log_std_ = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)

        dist = torch.distributions.Normal(mu, log_std_.exp())
        action = dist.rsample()

        if add_noise:
            noise = torch.normal(
                mean=torch.zeros_like(action),
                std=torch.full_like(action, self.epsilon_noise),
            )
            noise.clamp_(self.noise_clip_min, self.noise_clip_max)
            action += noise

        action.clamp_(self.min_action, self.max_action)

        return action, dist

    @torch.no_grad()
    def get_action(self, state, evaluation=False) -> np.ndarray:
        if state.ndim != 2:
            state = state[None, ...]

        state = torch.tensor(state, dtype=torch.float32, device=self.device)

        _action, prob = self.forward(state)

        noise = torch.normal(
            mean=torch.zeros_like(_action),
            std=torch.full_like(_action, self.epsilon_noise),
        )
        noise.clamp_(self.noise_clip_min, self.noise_clip_max)

        action = prob.mean if evaluation else prob.sample() + noise
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        q_val = self.q_net(state_action)

        return q_val


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
) -> TBR:
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

    controller = TBR(
        actor=actor,
        critic_1=critic_1,
        critic_2=critic_2,
        actor_optimizer=actor_optimizer,
        critic_optimizer_1=critic_1_optimizer,
        critic_optimizer_2=critic_2_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        lambda_td=config.lambda_td,
        alpha_start=config.alpha_start,
        alpha_finish=config.alpha_finish,
        epsilon_noise_offline=config.epsilon_noise_offline,
        epsilon_noise_online=config.epsilon_noise_online,
        online_budget=config.num_train_ops_online,
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
    controller: TBR,
    config: TrainConfig,
    mode: str,
):
    modes = ["offline_training", "offline_refinement", "online_training"]
    if mode not in modes:
        raise ValueError(
            "Mode should be either 'offline_training', 'offline_refinement' or 'online_training'"
        )

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
            f"{mode}/test_normalized_reward": normalized_reward.mean(),
            f"{mode}/test_reward": episode_rewards.mean(),
        },
        step=epoch,
    )

    wandb.log(
        {
            "eval/reward_std": episode_rewards.std(),
            "eval/normalized_reward_std": normalized_reward.std(),
        },
        step=epoch,
    )

    # save
    torch.save(
        controller.state_dict(),
        os.path.join(config.checkpoints_path, f"{mode}", f"tdbcrf_{mode}_{epoch}.pt"),
    )


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
    e_global = config.num_train_ops_offline + config.num_train_ops_policy_refinement

    if config.checkpoint_path_trained_offline is None:
        checkpoint_path = os.path.join(
            config.checkpoints_path,
            "offline_refinement",
            f"tdbcrf_offline_refinement_{e_global}.pt",
        )
    else:
        checkpoint_path = config.checkpoint_path_trained_offline

    td_bc_rf = create_controller(state_dim, action_dim, config)
    td_bc_rf.select_mode("online_training")
    td_bc_rf.load_state_dict(
        torch.load(checkpoint_path, map_location=torch.device(config.device))
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

    # populate buffer with current policy trajectories
    _ = collect_traj_from_env(envs, td_bc_rf.actor, buffer)

    for e in trange(config.num_train_ops_online):
        if (e + 1) % config.updates_before_collections == 0:
            episode_reward = collect_traj_from_env(envs, td_bc_rf.actor, buffer)
            wandb.log(
                {"online/train_episode_reward": episode_reward.mean()},
                step=(e_global + e + 1),
            )

        batch = buffer.sample(config.batch_size)
        update_info = td_bc_rf.update(batch)
        wandb.log(update_info, step=(e_global + e))

        if (e + 1) % config.eval_frequency == 0:
            evaluate(
                epoch=(e_global + e + 1),
                env_eval=env_eval,
                controller=td_bc_rf,
                config=config,
                mode=td_bc_rf.mode,
            )


def train_offline(
    env_eval: gym.Env,
    config: TrainConfig,
    dataset: Dict[str, np.ndarray],
    mode="offline_training",
):
    modes = ["offline_training", "offline_refinement", "online_training"]
    if mode not in modes:
        raise ValueError(
            "Mode should be either 'offline_training', 'offline_refinement' or 'online_training'"
        )

    state_dim = env_eval.observation_space.shape[0]
    action_dim = env_eval.action_space.shape[0]

    buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)
    buffer.load_d4rl_dataset(dataset)

    # create controller and set training mode
    td_bc_rf = create_controller(state_dim, action_dim, config)
    td_bc_rf.select_mode(mode)

    if mode == "offline_refinement":
        num_iter = config.num_train_ops_policy_refinement
        global_e = config.num_train_ops_offline
        if config.checkpoint_path_trained_offline is None:
            checkpoint_path = os.path.join(
                config.checkpoints_path,
                "offline_refinement",
                f"tdbcrf_offline_training_{global_e}.pt",
            )
        else:
            checkpoint_path = config.checkpoint_path_trained_offline

        td_bc_rf.load_state_dict(
            torch.load(checkpoint_path, map_location=torch.device(config.device))
        )
        print("checkpoint loaded")
    else:
        num_iter = config.num_train_ops_offline
        global_e = 0

    for e in trange(num_iter):
        log_step = global_e + e
        batch = buffer.sample(config.batch_size)
        update_info = td_bc_rf.update(batch)
        wandb.log(update_info, step=log_step)

        if (e + 1) % config.eval_frequency == 0:
            evaluate(
                epoch=(log_step + 1),
                env_eval=env_eval,
                controller=td_bc_rf,
                config=config,
                mode=td_bc_rf.mode,
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

    print(f"Checkpoints path: {config.checkpoints_path}")
    os.makedirs(config.checkpoints_path, exist_ok=True)
    os.makedirs(
        os.path.join(config.checkpoints_path, "offline_training"), exist_ok=True
    )
    os.makedirs(
        os.path.join(config.checkpoints_path, "offline_refinement"), exist_ok=True
    )
    os.makedirs(os.path.join(config.checkpoints_path, "online_training"), exist_ok=True)
    with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
        pyrallis.dump(config, f)

    if config.train_offline:
        print("\nStarting **offline** training...")
        train_offline(env, config, dataset, mode="offline_training")

    if config.train_refinement:
        train_offline(env, config, dataset, mode="offline_refinement")

    if config.train_online:
        print("\nStarting **online** training...")
        train_online(env, config, dataset, state_mean, state_std)

    wandb.finish()


if __name__ == "__main__":
    main()
