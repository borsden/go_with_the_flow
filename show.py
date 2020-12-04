import numpy as np
import tqdm

from env import Environment, Visualization

rng = np.random.default_rng()

TURNS = 100

AGENT_BIRTHS_PER_TURN = 100

RATIO_RANDOM_BIRTH = 0.1

MOVE_PREFERENCE_MATRIX = np.array([
    [
        [2, 2, 2],
        [1, 1, 1],
        [np.nan, np.nan, np.nan],
    ],
    [
        [2, 3, 2],
        [1, np.nan, 1],
        [np.nan, np.nan, np.nan],
    ],
    [
        [2, 2, 2],
        [1, 1, 1],
        [np.nan, np.nan, np.nan],
    ]
])  # The agent decides which space to move to by adding this move preference array to the value array of the surrounding environment.

MOVE_PROBABILITY_MATRIX = np.array([
    [
        [0.075, 0.075, 0.075],
        [0.0375, 0.0375, 0.0375],
        [0, 0, 0],
    ],
    [
        [0.075, 0.1, 0.075],
        [0.0375, 0, 0.0375],
        [0, 0, 0],
    ],
    [
        [0.075, 0.075, 0.075],
        [0.0375, 0.0375, 0.0375],
        [0, 0, 0],
    ]
])  # 10% of the time the agent moves randomly to an adjacent space. It is the move probability matrix.

# We want to have XYZ format for all matrices, so satisfy this condition
MOVE_PREFERENCE_MATRIX = np.moveaxis(MOVE_PREFERENCE_MATRIX, (0, 1, 2), (1, 2, 0))
MOVE_PROBABILITY_MATRIX = np.moveaxis(MOVE_PROBABILITY_MATRIX, (0, 1, 2), (1, 2, 0))

SEED = 500

ENV_NAME = "vol3"
ENV_ARR = np.load(f"./envs/{ENV_NAME}_environment.npy")
ACC_ARR = np.load(f"./envs/{ENV_NAME}_accumulations.npy")

env = Environment(ENV_ARR, seed=SEED)

for i in tqdm.tqdm(range(TURNS)):
    env.born(AGENT_BIRTHS_PER_TURN, RATIO_RANDOM_BIRTH)
    env.move(MOVE_PREFERENCE_MATRIX, MOVE_PROBABILITY_MATRIX)

TRACKED_AGENT_RATIO = 1.  # Ratio of agents tracked
visualization = Visualization(
    env, tracked_agents_ratio=TRACKED_AGENT_RATIO,
    # accumulation=ACC_ARR # Uncomment this, if you want to look at accumulations too."""
)

"""Uncomment this, if you want to look an latest positions."""
# visualization.show_locations()

"""Uncomment this, if you want to look at at tracks."""
# TODO(!!!!!!): Sadly right now we can't use opacity for every point in array.
#  (https://github.com/pyvista/pyvista/pull/855) after adding this, we can uncomment this code and it will look
#  prettier.
# visualization.show_tracks()
#
"""Uncomment this, if you want to look at animated movement."""
visualization.show_animated_movement()
