from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Type

import numpy as np
import torch
from gymnasium import spaces

from gym_agent import utils
from gym_agent.core.base.onpolicy import OnPolicyAlgorithm, OnPolicyConfig
from gym_agent.core.base.polices import ActorCriticPolicy
from gym_agent.core.buffers import RolloutBuffer
from gym_agent.core.callbacks import Callbacks
from gym_agent.core.types import ActType, ObsType


@dataclass(kw_only=True)
class ActorCriticConfig(OnPolicyConfig):
    gamma: float = 0.99
    gae_lambda: float = 1.0
    rollout_buffer_cls: Optional[type[RolloutBuffer]] = RolloutBuffer
    rollout_buffer_kwargs: Optional[dict] = None


class OnPolicyActorCriticAlgorithm(OnPolicyAlgorithm[ObsType, ActType]):
    """This is a base class for actor-critic agents.
    You can consider this class as an on-policy agent with a value function.
    """

    memory: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        env_id: str,
        policy: ActorCriticPolicy,
        config: ActorCriticConfig = None,
        supported_action_spaces: Optional[tuple[type[spaces.Space], ...]] = None,
    ):
        config = config if config is not None else ActorCriticConfig()
        super().__init__(
            env_id=env_id,
            policy=policy,
            config=config,
            supported_action_spaces=supported_action_spaces,
        )

        rollout_buffer_kwargs = {
            "buffer_size": config.n_steps,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "device": self.device,
            "num_envs": self.num_envs,
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda,
            "seed": self.seed,
        } | (
            config.rollout_buffer_kwargs
            if config.rollout_buffer_kwargs is not None
            else {}
        )

        self.memory = config.rollout_buffer_cls(**rollout_buffer_kwargs)

        self._last_obs = None

    @abstractmethod
    def predict(self, state: ObsType, deterministic: bool = True) -> ActType:
        pass

    @abstractmethod
    def learn(self) -> None:
        """
        Perform learning using the experiences stored in the memory buffer.

        This method should be overridden by subclasses to implement specific learning algorithms.
        The method should utilize the experiences stored in `self.memory` to update the agent's policy.
        use self.memory.get(batch_size) to get a generator that yields batches of experiences.
        """
        raise NotImplementedError

    def evaluate_actions(
        self, obs: ObsType, actions: ActType
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions using the current policy.

        Args:
            obs (ObsType): The input observations which can be either a numpy array or a dictionary
                * ``NDArray`` shape - `[batch, *obs_shape]`
                * ``dict`` shape - `[key: sub_space]` with `sub_space` shape: `[batch, *sub_obs_shape]`
            actions (ActType): The actions to evaluate which can be either a numpy array or a dictionary
                * ``NDArray`` shape - `[batch, *action_shape]`
                * ``dict`` shape - `[key: sub_space]` with `sub_space` shape: `[batch, *sub_action_shape]`
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - values (torch.Tensor): The value estimates for the observations.
                - action_log_probs (torch.Tensor): The log probabilities of the actions.
                - entropy (torch.Tensor): The entropy of the action distribution.
        """
        action_logits, value_logits = self.policy.forward(obs)
        values = value_logits.squeeze(-1)

        distribution = self.distribution(action_logits)
        action_log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, action_log_probs, entropy

    def collect_buffer(self, deterministic: bool, callbacks: Type[Callbacks] = None):
        # TODO: this is not an episode, so the callbacks should be different
        callbacks.on_episode_begin()

        self.memory.reset()

        time_steps = 0
        episodes = 0

        self.policy.train()  # ensure in train mode
        for _ in range(self.n_steps):
            time_steps += 1

            with torch.no_grad():
                action_logits, value_logits = self.policy.forward(
                    utils.to_torch(self._last_obs, self.device)
                )

                distribution = self.distribution(action_logits)

                # get the action
                if deterministic:
                    actions = distribution.mode()
                else:
                    actions = distribution.sample()

                log_probs = distribution.log_prob(actions).cpu().numpy()
                actions = actions.cpu().numpy()
                values = value_logits.squeeze(-1).cpu().numpy()

            next_obs, rewards, terminated, truncated, info = self.envs.step(actions)

            self.memory.add(
                self._last_obs,
                actions,
                rewards,
                values,
                log_probs,
                self._last_episode_starts,
            )

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

        # compute the value of the last observation
        with torch.no_grad():
            values = (
                self.policy.forward_critic(utils.to_torch(self._last_obs, self.device))
                .squeeze(-1)
                .cpu()
                .numpy()
            )

        self.memory.calc_advantages_and_returns(
            last_values=values, last_terminals=self._last_episode_starts
        )

        # TODO: this is not an episode, so the callbacks should be different
        callbacks.on_episode_end()

        return time_steps, int(episodes)
