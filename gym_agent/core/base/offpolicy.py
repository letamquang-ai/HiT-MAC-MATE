from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Type

import numpy as np
import torch
from gymnasium import spaces

from gym_agent.core.types import ActType, ObsType


from gym_agent.core.base.base_algo import (
    BaseAlgorithm,
    BaseConfig,
)
from gym_agent.core.buffers import ReplayBuffer
from gym_agent.core.callbacks import Callbacks
from gym_agent.core.base.polices import BasePolicy


@dataclass(kw_only=True)
class OffPolicyConfig(BaseConfig):
    gamma: float = 0.99
    buffer_size: int = int(1e5)
    replay_buffer_cls: Optional[type[ReplayBuffer]] = ReplayBuffer
    replay_buffer_kwargs: Optional[dict] = None


class OffPolicyAlgorithm(BaseAlgorithm[ObsType, ActType]):
    memory: ReplayBuffer

    def __init__(
        self,
        env_id: str,
        policy: BasePolicy,
        config: OffPolicyConfig = None,
        supported_action_spaces: Optional[tuple[type[spaces.Space], ...]] = None,
    ):
        config = config if config is not None else OffPolicyConfig()
        super().__init__(
            env_id=env_id,
            policy=policy,
            config=config,
            supported_action_spaces=supported_action_spaces,
        )

        self.gamma = config.gamma

        replay_buffer_kwargs = {
            "buffer_size": config.buffer_size,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "device": self.device,
            "num_envs": self.num_envs,
            "seed": self.seed,
        } | (
            config.replay_buffer_kwargs
            if config.replay_buffer_kwargs is not None
            else {}
        )

        self.memory = config.replay_buffer_cls(**replay_buffer_kwargs)

    def collect_buffer(
        self, deterministic: bool, callbacks: Type[Callbacks]
    ) -> tuple[int, int]:
        # TODO: this is not an episode, so the callbacks should be different
        callbacks.on_episode_begin()

        time_steps = 0
        episodes = 0

        self.policy.train()  # ensure in train mode
        for _ in range(self.n_steps):
            time_steps += 1

            with torch.no_grad():
                actions = self.predict(self._last_obs, deterministic)
            next_obs, rewards, terminated, truncated, info = self.envs.step(actions)

            self.memory.add(self._last_obs, actions, rewards, terminated)

            self._last_obs = next_obs
            self._last_episode_starts = np.array(
                terminated | truncated
            )  # episode starts is just done

            self.current_scores += np.array(rewards, dtype=np.float32)

            # if an env is done, record the score and reset the current score for that env
            self.scores.extend(self.current_scores[self._last_episode_starts].tolist())
            # reset the current score for that env
            self.current_scores[self._last_episode_starts] = 0.0
            # count how many episodes finished
            episodes += np.sum(self._last_episode_starts)

        # TODO: this is not an episode, so the callbacks should be different
        callbacks.on_episode_end()

        return time_steps, int(episodes)

    @abstractmethod
    def learn(
        self,
    ) -> None:
        """
        Abstract method to be implemented by subclasses for learning from a batch of experiences.
        use self.memory.sample(self.batch_size) to get a batch of experiences.
        """
        ...
