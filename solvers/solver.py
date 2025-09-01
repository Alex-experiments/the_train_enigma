from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterable


class Actions(Enum):
    """Enumeration of possible actions that can be taken in the solver.

    Attributes:
        MOVE_RIGHT: Move the solver to the right.
        MOVE_LEFT: Move the solver to the left.
        SWITCH_LIGHT: Switch the light in the current wagon.
        END: End the solving process.
    """

    MOVE_RIGHT = "move_right"
    MOVE_LEFT = "move_left"
    SWITCH_LIGHT = "switch_light"
    END = "end"


class Solver(ABC):
    """Abstract base class for solvers.

    Attributes:
        wagon_count: The number of wagons in the problem.
            Must be set just before return Actions.END
    """

    def __init__(self):
        """Initializes the Solver with a default wagon count of 0."""
        self.wagon_count = 0

    @abstractmethod
    def get_next_step(self, cur_wagon_state: bool) -> Iterable[Actions]:
        """Gets the next step in the solving process.

        Args:
            cur_wagon_state (bool): The current state of the wagon.
                - True if the wagon is light up, False otherwise.

        Returns:
            Iterable[Actions]: An iterable over the list of actions to be taken next.
        """
