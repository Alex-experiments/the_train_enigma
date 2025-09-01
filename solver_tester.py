import importlib.util
import inspect
import os
import time
import tracemalloc
from argparse import ArgumentParser
from typing import Optional, Type

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D
from tqdm import tqdm

from solvers.solver import Actions, Solver


class SolverTester:
    """A class to test and visualize the performance of a solver on a train configuration.

    Attributes:
        wagons_states (np.ndarray): Array representing the current state of the train.
        n_wagons (int): Number of wagons in the train.
        solver (Solver): The solver instance being tested.
        viz (bool): Flag indicating whether to visualize the solver's actions.
        viz_pause (float): Time to pause between visualization steps.
        solver_pos (int): Current position of the solver in the train.
        n_solver_actions (int): Total number of actions performed by the solver.
        n_solver_move_actions (int): Number of move actions performed by the solver.
        ax (matplotlib.axes.Axes): Matplotlib axes for visualization.
        angles (np.ndarray): Array of angles for positioning wagons in visualization.
        solver_pos_radius (float): Radius for positioning the solver in visualization.
        solver_pos_patch (matplotlib.patches.Circle): Circle patch representing the solver's
            position.
    """

    def __init__(
        self,
        wagons_states: np.ndarray,
        solver: Solver,
        viz: bool = False,
        viz_fps: float = 5,
    ):
        self.wagons_states = wagons_states
        self.n_wagons = wagons_states.shape[0]
        self.solver = solver
        self.viz = viz
        self.viz_pause = 1.0 / viz_fps

        self.solver_pos = 0
        self.n_solver_actions = 0
        self.n_solver_move_actions = 0

        if self.viz:
            self.init_viz()

    def check_solver(self, max_steps: Optional[int] = None) -> bool:
        """Checks if the solver can solve the given train configuration.

        Args:
            max_steps (Optional[int]): Maximum number of steps allowed for the solver.
                If None, the solver can run indefinitely.

        Returns:
            bool: True if the solver successfully solved the train configuration,
                False otherwise.
        """
        if self.viz:
            plt.pause(self.viz_pause)

        while max_steps is None or self.n_solver_actions < max_steps:
            # Get the solver actions
            actions = self.solver.get_next_step(
                cur_wagon_state=self.wagons_states[self.solver_pos]
            )

            # Iterate over them
            for action in actions:
                self.n_solver_actions += 1

                match action:
                    case Actions.MOVE_RIGHT:
                        self.solver_pos = (self.solver_pos + 1) % self.n_wagons
                        self.n_solver_move_actions += 1
                        if self.viz:
                            self.update_solver_pos_on_viz()
                    case Actions.MOVE_LEFT:
                        self.solver_pos = (self.solver_pos - 1) % self.n_wagons
                        self.n_solver_move_actions += 1
                        if self.viz:
                            self.update_solver_pos_on_viz()
                    case Actions.SWITCH_LIGHT:
                        self.wagons_states[self.solver_pos] = not self.wagons_states[
                            self.solver_pos
                        ]
                        if self.viz:
                            self.ax.patches[self.solver_pos].set_facecolor(
                                "white"
                                if self.wagons_states[self.solver_pos]
                                else "black"
                            )
                    case Actions.END:
                        if self.viz:
                            plt.close()
                        return self.solver.wagon_count == self.n_wagons

                if self.viz:
                    if self.n_solver_actions < 50:
                        plt.savefig(f"images/{self.n_solver_actions}.png")
                    else:
                        break
                    plt.pause(self.viz_pause)

        # Instead of raising an error, just return a failure case
        print("MaxIteration stop")
        if self.viz:
            plt.close()
        return False

    def update_solver_pos_on_viz(self) -> None:
        """Updates the position of the solver in the visualization."""
        self.solver_pos_patch.center = (
            self.solver_pos_radius * np.cos(self.angles[self.solver_pos]),
            self.solver_pos_radius * np.sin(self.angles[self.solver_pos]),
        )

    def init_viz(self) -> None:
        """Initializes the visualization for the solver.

        Wagons (and the solver pos) are created as patches, so that they can
        easily be modified later without recreating the whole plot from scratch.
        """
        _, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        rect_width = 0.5
        rect_height = 1
        radius = rect_width * self.n_wagons / np.pi

        # Place wagons in a circular manner
        self.angles = np.linspace(
            -0.5 * np.pi, 1.5 * np.pi, self.n_wagons, endpoint=False
        )

        for i, angle in enumerate(self.angles):
            x = radius * np.cos(angle)
            y = radius * np.sin(angle) - rect_height / 2

            # Wagons with light on will be white, others will be black
            rect = Rectangle(
                (x, y),
                rect_width,
                rect_height,
                facecolor="white" if self.wagons_states[i] else "black",
                edgecolor="k",
            )

            # Rotate rectangle around its center
            t = (
                Affine2D().rotate_around(
                    radius * np.cos(angle), radius * np.sin(angle), angle
                )
                + self.ax.transData
            )
            rect.set_transform(t)

            self.ax.add_patch(rect)

        # Display the solver position as a red circle
        self.solver_pos_radius = radius + 2 * rect_width
        self.solver_pos_patch = Circle(
            (0, 0), radius=0.1, facecolor="red", edgecolor="k"
        )
        self.update_solver_pos_on_viz()
        self.ax.add_patch(self.solver_pos_patch)

        # Set plot limits to ensure all rectangles are visible
        self.ax.set_xlim(-self.solver_pos_radius - 0.5, self.solver_pos_radius + 0.5)
        self.ax.set_ylim(-self.solver_pos_radius - 0.5, self.solver_pos_radius + 0.5)


def get_solver_subclass(solver_file_path: str) -> Type[Solver]:
    """Retrieves the solver subclass from the given file path.

    Args:
        solver_file_path (str): Path to the file containing the solver subclass.

    Returns:
        Type[Solver]: The solver subclass found in the file.

    Raises:
        ValueError: If no class inheriting from Solver is found in the file.
    """
    # Load the module from the given file path
    spec = importlib.util.spec_from_file_location("solvermodule", solver_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Iterate over the module's attributes to find the subclass of Solver
    for _, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, Solver):
            return obj

    raise ValueError(
        f"No class inheriting from Solver has been found in {solver_file_path}."
    )


def test_subset(folder_path: str, solver_class: Type[Solver]) -> None:
    """Tests the solver on a subset of test cases.

    Args:
        folder_path (str): Path to the folder containing the test cases.
        solver_class (Type[Solver]): The solver to be tested.
    """
    cumulated_solving_time = 0
    peak_memory_usage = 0

    cumulated_solver_actions = 0
    cumulated_solver_move_actions = 0

    # Get all test cases from that subset
    for filename in tqdm(os.listdir(folder_path)):
        if not filename.endswith(".npy"):
            continue

        file_path = os.path.join(folder_path, filename)

        with open(file_path, "rb") as f:
            wagons_states = np.load(f)

        tester = SolverTester(
            wagons_states=wagons_states, solver=solver_class(), viz=False
        )

        start_time = time.perf_counter()
        tracemalloc.start()

        solved = tester.check_solver(max_steps=None)

        end_time = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if not solved:
            raise ValueError(
                f"The solver failed on the following test case {file_path}"
            )

        cumulated_solving_time += end_time - start_time
        peak_memory_usage = max(peak_memory_usage, peak)

        cumulated_solver_actions += tester.n_solver_actions
        cumulated_solver_move_actions += tester.n_solver_move_actions

    print(f"Total solving time: {cumulated_solving_time:.6f} seconds")
    print(f"Peak memory usage: {peak_memory_usage} bytes")
    print(
        f"Solved all test cases in {cumulated_solver_actions:,} actions, "
        f"among which {cumulated_solver_move_actions:,} where move actions"
    )


if __name__ == "__main__":

    parser = ArgumentParser(description="Check solver.")
    parser.add_argument(
        "-fp",
        "--solver_file_path",
        type=str,
        required=True,
        help="Path to the file containing the solver subclass.",
    )
    parser.add_argument(
        "-t",
        "--test_subset",
        type=str,
        choices=["small", "medium", "big"],
        default="small",
        help="Which test subset to run.",
    )
    args = parser.parse_args()

    test_subset(
        os.path.join("test_cases", f"{args.test_subset}_trains"),
        solver_class=get_solver_subclass(solver_file_path=args.solver_file_path),
    )
