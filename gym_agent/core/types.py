from typing import TypeVar
from numpy.typing import NDArray


ObsType = TypeVar("ObsType", NDArray, dict[str, NDArray])
ActType = TypeVar("ActType", NDArray, dict[str, NDArray])

