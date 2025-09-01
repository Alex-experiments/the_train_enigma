from argparse import ArgumentParser

import numpy as np

from solver_tester import SolverTester, get_solver_subclass


def vizualise_solver_run(
    solver_file_path: str,
    wagon_states: np.ndarray,
    viz_fps: float,
) -> None:
    """Visualizes the solver's run on a given train configuration.

    Args:
        solver_file_path (str): Path to the file containing the solver subclass.
        wagon_states (np.ndarray): Array representing the initial state of the train.
        viz_fps (float): Visualization frames per second.
    """

    solver = get_solver_subclass(solver_file_path=solver_file_path)
    solver_instance = solver()

    solved = SolverTester(
        wagons_states=wagon_states,
        solver=solver_instance,
        viz=True,
        viz_fps=viz_fps,
    ).check_solver(max_steps=None)

    if solved:
        print("The solver ended and found the correct solution.")
    else:
        print(
            "The solver ended and failed to find the correct solution. "
            f"Instead of finding {wagon_states.shape[0]}, he found {solver_instance.wagon_count}"
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
        "--train_file_path",
        type=str,
        default=None,
        help="Specify the path to a .npy file if you don't want to test on a random train.",
    )
    parser.add_argument(
        "-n",
        "--n_wagons",
        type=int,
        default=20,
        help="The size of the train. Only used for random train generation.",
    )
    parser.add_argument(
        "-fps",
        "--viz_fps",
        type=float,
        default=24.0,
        help="The animation speed (frame per second).",
    )
    args = parser.parse_args()

    if args.train_file_path is None:
        arr = np.random.choice([False, True], size=args.n_wagons)
    else:
        with open(args.train_file_path, "rb") as f:
            arr = np.load(f)

    vizualise_solver_run(
        solver_file_path=args.solver_file_path,
        wagon_states=arr,
        viz_fps=args.viz_fps,
    )
