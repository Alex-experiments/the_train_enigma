from typing import Iterable
from solvers.solver import Actions, Solver


class MySolver(Solver):
    def get_next_step(self, cur_wagon_state: bool) -> Iterable[Actions]:
        """Gets the next step in the solving process.

        Args:
            cur_wagon_state (bool): The current state of the wagon.
                - True if the wagon is light up, False otherwise.

        Returns:
            Iterable[Actions]: An iterable over the list of actions to be taken next.
        """

        # TODO (Remember that self.wagon_count must be set before returning END)
