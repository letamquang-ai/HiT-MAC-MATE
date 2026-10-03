from dataclasses import dataclass

from gym_agent.core.base.base_algo import BaseAlgorithm, BaseConfig

from gym_agent.core.types import ActType, ObsType


@dataclass(kw_only=True)
class OnPolicyConfig(BaseConfig):
    pass  # no additional parameters for now


class OnPolicyAlgorithm(BaseAlgorithm[ObsType, ActType]):
    """
    This is a base class for on-policy agents.
    Because there are on-policy agents that are not actor-critic, so this is a placeholder for the future on-policy implementations if they arise.
    """

    pass
