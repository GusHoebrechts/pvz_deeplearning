import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from sb3_contrib.common.maskable.buffers import (
    MaskableDictRolloutBuffer,
    MaskableRolloutBuffer,
)
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import (
    get_action_masks,
    is_masking_supported,
)


class MaskedA2C(A2C):
    """A2C variant that applies action masks during collection and updates."""

    policy_aliases = {"MlpPolicy": MaskableActorCriticPolicy}

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.use_sde:
            raise ValueError(
                "MaskedA2C does not support generalized state-dependent exploration."
            )

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        ).to(self.device)
        self._get_maskable_policy()

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = MaskableDictRolloutBuffer
            else:
                self.rollout_buffer_class = MaskableRolloutBuffer
        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self._maybe_recommend_cpu()

    def _get_maskable_policy(self) -> MaskableActorCriticPolicy:
        if not isinstance(self.policy, MaskableActorCriticPolicy):
            raise TypeError("MaskedA2C requires MaskableActorCriticPolicy.")
        return self.policy

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        if not isinstance(
            rollout_buffer, (MaskableRolloutBuffer, MaskableDictRolloutBuffer)
        ):
            raise TypeError("MaskedA2C requires a mask-aware rollout buffer.")
        if not is_masking_supported(env):
            raise ValueError("The training environment must provide action_masks().")
        if self._last_obs is None:
            raise RuntimeError("No previous observation was provided.")
        if self._last_episode_starts is None:
            raise RuntimeError("No previous episode-start flags were provided.")

        policy = self._get_maskable_policy()
        last_obs = self._last_obs
        last_episode_starts = self._last_episode_starts
        policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            action_masks = get_action_masks(env)
            with th.no_grad():
                obs_tensor = obs_as_tensor(last_obs, self.device)
                actions, values, log_probs = policy(
                    obs_tensor, action_masks=action_masks
                )
            actions = actions.cpu().numpy()

            new_obs, rewards, dones, infos = env.step(actions)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                last_obs,
                actions,
                rewards,
                last_episode_starts,
                values,
                log_probs,
                action_masks=action_masks,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones
            last_obs = new_obs
            last_episode_starts = dones

        with th.no_grad():
            values = policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones
        )

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic=False,
        action_masks=None,
    ):
        if action_masks is None:
            raise ValueError("MaskedA2C.predict requires action_masks.")
        return self._get_maskable_policy().predict(
            observation,
            state,
            episode_start,
            deterministic,
            action_masks=action_masks,
        )

    def train(self):
        policy = self._get_maskable_policy()
        policy.set_training_mode(True)
        self._update_learning_rate(policy.optimizer)

        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.long().flatten()

            values, log_prob, entropy = policy.evaluate_actions(
                rollout_data.observations,
                actions,
                action_masks=rollout_data.action_masks,
            )
            values = values.flatten()
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

            policy_loss = -(advantages * log_prob).mean()
            value_loss = th.nn.functional.mse_loss(rollout_data.returns, values)
            if entropy is None:
                entropy_loss = th.mean(log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            loss = (
                policy_loss
                + self.ent_coef * entropy_loss
                + self.vf_coef * value_loss
            )

            policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(
                policy.parameters(), self.max_grad_norm
            )
            policy.optimizer.step()

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())


class A2CMaskWrapper(gym.Wrapper):
    """Normalize PvZ observations and expose legal-action masks to A2C."""

    def __init__(self, env):
        super().__init__(env)
        base_env = self.env.unwrapped
        self._pvz_observation_layout = None

        if hasattr(base_env, "plant_deck"):
            n_plants = len(base_env.plant_deck)
            observation_size = self.observation_space.shape[0]
            grid_size = (observation_size - 1 - 2 * n_plants) // 5
            if (
                grid_size <= 0
                or 5 * grid_size + 1 + 2 * n_plants != observation_size
            ):
                raise ValueError("Unexpected PvZ observation layout.")

            max_plant_hp = max(
                plant.MAX_HP for plant in base_env.plant_deck.values()
            )
            if max_plant_hp <= 0:
                raise ValueError("PvZ plant hit points must be positive.")

            self._pvz_observation_layout = (
                grid_size,
                n_plants,
                max_plant_hp,
            )
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=self.observation_space.shape,
                dtype=np.float32,
            )

    def _normalize_observation(self, observation):
        if self._pvz_observation_layout is None:
            return observation

        grid_size, n_plants, max_plant_hp = self._pvz_observation_layout
        normalized = np.asarray(observation, dtype=np.float32).copy()
        normalized[:grid_size] /= n_plants
        normalized[grid_size : 2 * grid_size] /= max_plant_hp
        normalized[2 * grid_size : 3 * grid_size] = (
            np.log1p(normalized[2 * grid_size : 3 * grid_size])
            / np.log1p(10_000)
        )
        normalized[3 * grid_size : 4 * grid_size] = (
            np.log1p(normalized[3 * grid_size : 4 * grid_size])
            / np.log1p(10)
        )
        sun_index = 5 * grid_size
        normalized[sun_index] = (
            np.log1p(normalized[sun_index]) / np.log1p(10_000)
        )
        return np.clip(normalized, 0.0, 1.0)

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        return self._normalize_observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            self._normalize_observation(observation),
            reward,
            terminated,
            truncated,
            info,
        )

    def action_masks(self) -> np.ndarray:
        return np.asarray(
            self.env.unwrapped.mask_available_actions(), dtype=bool
        )
