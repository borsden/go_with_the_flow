import pandas as pd
import numpy as np
import cupy as cp
import pyvista as pv
import tqdm

def cupy_ravel_multi_index(multi_index, dims):
    """
    Sadly  CuPy 7.6.0 does not have ravel_multi_index function (only from 8 version)
    Implement simple version of it.

    :param multi_index: multi indexes
    :param dims: dimension of target
    """
    ndim = len(dims)

    s = 1
    ravel_strides = [1] * ndim

    for i in range(ndim - 2, -1, -1):
        s = s * dims[i + 1]
        ravel_strides[i] = s

    multi_index = cp.broadcast_arrays(*multi_index)
    raveled_indices = cp.zeros(multi_index[0].shape, dtype=cp.int64)
    for d, stride, idx in zip(dims, ravel_strides, multi_index):
        idx = idx.astype(cp.int64, copy=False)

        raveled_indices += stride * idx
    return raveled_indices


cp.ravel_multi_index = cupy_ravel_multi_index


def neighbour_positions_delta():
    """
    3x3x3x3 array, where 0,1,2 axes is X,Y,Z respectively and third axis is 3 values, which we should
    add to index of center cell to receive an index for this cell.
    Example: if center cell is [5, 2, 3], than left-deep-upper corner will be [5, 2, 3] + [-1, +1, -1] = [4, 3, 2],
    where [-1, -1, +1] is a value from this array.
    """
    neighbour_positions_delta = np.array([-1, 0, 1])
    neighbour_positions_delta = np.array(np.meshgrid(*[neighbour_positions_delta] * 3, indexing='ij')).T
    neighbour_positions_delta = neighbour_positions_delta.swapaxes(0, 2)
    return cp.asarray(neighbour_positions_delta)


NEIGHBOUR_POSITIONS_DELTA = neighbour_positions_delta()


class Environment:
    def __init__(self, env_arr, seed=300):
        """Initialize environment.

        Pad env with NaN in every axis. It will help us to find neighbours of cells even if they are edges.

        env should have 3 dimensions (XYZ).

        {_agents_positions} is an array of current XYZ position for every agent.
        {agents_state} is an array of agents states. 1 - live, 0 - dead
        {is_available_env} - env-like boolean array, where True indicates free cell

        :param env_arr: array of environment, where agents born. move and die.
        """
        self._raw_env = env_arr
        self.env = cp.pad(cp.array(env_arr).astype(cp.float), 1, mode='constant', constant_values=cp.NaN)

        self.agents_state = None
        self._agents_positions = None

        self._agents_positions_all_time = []

        self.is_available_env = ~cp.isnan(self.env)

        self._rng = np.random.default_rng(seed)
        np.random.seed(seed)
        cp.random.seed(seed)

    def born(self, number_of_agents, ratio_random_birth):
        """
        Method, which born new agents.
        1.  All new agents should be borned in free cells.
        2.  {BORN_IN_BOTTOM} agents are born at the bottom of ENV.
            There can be hypothetical situations, when we already have died agents in bottom cells.
            To avoid this problem, we filter this cells.
        3.  Generate an indexes of borned agents in envs.
            !!!!!WARNING!!!!!: Perhaps situation, when we have less available cells than number of new agents
            (ex.: if we have 10x10x100 env and want to create 200 agents at bottom).
            We can simple handle it, setting number of new agents like min(free_cells, number_of_agents).
        4.  The remaining agents should not appear into already occupied positions.
            {agent_available_env_bottom} is just a view, really we change {agent_available_env} array.
        5. Receive X,Y,Z positions of borned agents. Specify Z manually, because we now, that it is a bottom.
        6.  Other agents should be borned randomly in the whole envs.
            It is too slow to sample from whole {is_available_env}. So, use simple hack - because envs are
            much bigger, than number of agents - let's generate some random indexes there and just select free.
            Todo: Strictly, it can give us problems in some cases, when there will be too many agents,
            but don't worry about it now.

        7. All agents, which were born on the top will die immediately.
        8. Combine all new agents with others.

        :param number_of_agents: number of agents born each turn
        :param ratio_random_birth: ratio of agent birth locations (i.e. base of environment vs. random)
        """
        # (1)
        # (2)
        born_in_bottom = int(number_of_agents * (1 - ratio_random_birth))
        agent_available_env_bottom = self.is_available_env[:, :, -2]
        available_flat_bottom_positions = cp.flatnonzero(agent_available_env_bottom == True)
        # (3)
        selected_flat_bottom_positions = cp.random.choice(
            available_flat_bottom_positions,
            born_in_bottom,
            replace=False
        )
        # (4)
        self.is_available_env[:, :, -2].ravel()[selected_flat_bottom_positions] = False
        # (5)
        bottom_agents_positions = cp.unravel_index(
            selected_flat_bottom_positions,
            (*agent_available_env_bottom.shape, 1)
        )
        bottom_agents_positions = cp.vstack(bottom_agents_positions)
        bottom_agents_positions[2] = (self.is_available_env.shape[2] - 2)

        # (6)
        born_in_random = number_of_agents - born_in_bottom
        random_positions = cp.array([
            # Use numpy function, because it is faster.
            np.random.randint(1, ax_shape - 1, born_in_random * 4)
            for ax_shape in self.is_available_env.shape
        ])
        random_flat_positions = cp.ravel_multi_index(random_positions, self.is_available_env.shape)

        is_available = self.is_available_env.ravel()[random_flat_positions]

        selected_flat_uniform_positions = random_flat_positions[is_available][:born_in_random]
        uniform_agents_positions = cp.unravel_index(selected_flat_uniform_positions, self.is_available_env.shape)
        uniform_agents_positions = cp.vstack(uniform_agents_positions)
        # Todo: This code is correct, but too slow. Replace it with code above.

        # available_flat_uniform_positions = cp.flatnonzero(self.is_available_env)
        # selected_flat_uniform_positions = cp.random.choice(
        #     available_flat_uniform_positions,
        #     number_of_agents - born_in_bottom,
        #     replace=False
        # )
        # uniform_agents_positions = cp.unravel_index(selected_flat_uniform_positions, self.is_available_env.shape)
        # uniform_agents_positions = cp.vstack(uniform_agents_positions)

        # (7)
        new_agent_positions = cp.hstack([uniform_agents_positions, bottom_agents_positions]).T
        new_agent_state = (new_agent_positions[:, 2] != 1).astype(cp.bool)

        # (8)
        if self._agents_positions is None:
            self._agents_positions = new_agent_positions
            self.agents_state = new_agent_state
        else:
            self._agents_positions = cp.vstack([self._agents_positions, new_agent_positions])
            self.agents_state = cp.hstack([self.agents_state, new_agent_state])

        self.is_available_env.ravel()[self.agents_flat_positions] = False

    @property
    def agents_flat_positions(self):
        """Return flat indexes in env for all agents."""
        if self._agents_positions is None:
            return []
        return cp.ravel_multi_index(self._agents_positions.T, self.env.shape)

    # @property
    # def agents_neighbour_flat_positions(self):
    #     """
    #     For any agent we find 3x3x3 array of their neighbour cells indexes.
    #     Flatten position in env of an agent will be at the center of this array.
    #     Result array will be 4 dimensional.
    #     """
    #
    #     # faster version of np.stack([self._agents_positions] * 3 * 3 * 3, axis=-1)
    #     neighbour_positions = np.resize(self._agents_positions, (27, *self._agents_positions.shape))
    #     neighbour_positions = np.moveaxis(neighbour_positions, (0, 1, 2), (2, 0, 1))
    #
    #     neighbour_positions = neighbour_positions.reshape((-1, 3, 3, 3, 3)).swapaxes(1, 4)
    #     neighbour_positions = neighbour_positions + self.neighbour_positions_delta
    #
    #     neighbour_flattened_poses = np.ravel_multi_index(neighbour_positions.reshape(-1, 3).T, self.env.shape)
    #     return neighbour_flattened_poses.reshape((-1, 3, 3, 3))

    @property
    def agents_neighbour_flat_positions(self):
        """
        For any agent we find 3x3x3 array of their neighbour cells indexes.
        Flatten position in env of an agent will be at the center of this array.
        Result array will be 4 dimensional.
        """
        agents_positions = cp.asarray(self._agents_positions)

        neighbour_positions = cp.expand_dims(agents_positions, 0).repeat(27, axis=0)
        neighbour_positions = cp.moveaxis(neighbour_positions, (0, 1, 2), (2, 0, 1))
        neighbour_positions = neighbour_positions.reshape((-1, 3, 3, 3, 3)).swapaxes(1, 4)
        neighbour_positions = neighbour_positions + NEIGHBOUR_POSITIONS_DELTA
        neighbour_positions = neighbour_positions.reshape(-1, 3).T
        neighbour_flattened_poses = cp.ravel_multi_index(neighbour_positions, self.env.shape)
        return neighbour_flattened_poses.reshape((-1, 3, 3, 3))

    def move(self, move_preference_matrix, move_probability_matrix, ratio_random_move=0.1):
        """
        1.  Select all living agents and their neighbours.
        2.  Create a movement matrix. All occupied by agent cells should be unavailable for move.
            add {move_preference_matrix} for values of neighbours.
        3.  If agent does not have any available cells for moving - it should die.
            Drop all died agents from current moving agents.
        4.  10% of the time the agent moves randomly.
            Agent can't go to unavailable cells, so we recalculate probability for available neighbours.
            (sum of prob should be 1).
        5.  Vectorized way to get random indices from array of probs. Like random.choice, but for 2d array.
        6.  Find new flat indexes for random moving agents.
        7.  Find new flat indexes for normal moving agents. Before argmax selection we shuffle neighbours,
            otherwise we will use always first max index.
        8.  Create an array with new agents positions.
        9.  If two agents want to occupy same cell - then we accept only first.
            All agents, which was declined to move because of collision will die.
        10. If agent reach top - it dies too.


        :param move_preference_matrix:  The agent decides which space to move to by adding this move
                                        preference array to the value array of the surrounding environment.

        :param move_probability_matrix:  10% of the time the agent moves randomly to an adjacent space.
                                         It is the move probability matrix.
        :return:
        """
        # (1)
        live_agents_neighbour_flat_positions = self.agents_neighbour_flat_positions[self.agents_state]
        # (2)
        move_candidates = self.env.ravel()[live_agents_neighbour_flat_positions].copy()

        is_available = self.is_available_env.ravel()[live_agents_neighbour_flat_positions]
        move_candidates[~is_available] = cp.nan
        move_candidates = move_candidates + cp.asarray(move_preference_matrix)

        # (3)
        should_die = cp.all(cp.isnan(move_candidates.reshape(-1, 27)), axis=1)
        should_die_agents = cp.flatnonzero(self.agents_state)[should_die]

        self.agents_state[should_die_agents] = False

        move_candidates = move_candidates[~should_die]
        live_agents_neighbour_flat_positions = live_agents_neighbour_flat_positions[~should_die]

        # (4)
        is_random_move = cp.random.binomial(1, ratio_random_move, live_agents_neighbour_flat_positions.shape[0])
        is_random_move = is_random_move.astype(cp.bool)
        random_move_candidates = move_candidates[is_random_move]

        random_move_probs = (~cp.isnan(random_move_candidates) * cp.asarray(move_probability_matrix)).reshape(-1, 27)
        random_move_probs /= random_move_probs.sum(axis=1)[:, None]

        # (5)
        random_vals = cp.expand_dims(cp.random.rand(random_move_probs.shape[0]), axis=1)
        random_indexes = (random_move_probs.cumsum(axis=1) > random_vals).argmax(axis=1)

        # (6)
        random_live_agents_neighbour_flat_positions = live_agents_neighbour_flat_positions[is_random_move]
        random_new_positions = cp.take_along_axis(
            random_live_agents_neighbour_flat_positions.reshape(-1, 27),
            random_indexes[:, None], axis=1
        ).T[0]

        # (7)
        normal_move_candidates = move_candidates[~is_random_move]

        # normal_move_indexes = cp.nanargmax(normal_move_candidates.reshape(-1, 27), axis=1)[:, None]
        # smart analog of cp.nanargmax(normal_move_candidates.reshape(-1, 27), axis=1)[:, None]

        normal_flattened_move_candidates = normal_move_candidates.reshape(-1, 27)
        normal_shuffled_candidates_idx = cp.random.rand(*normal_flattened_move_candidates.shape).argsort(axis=1)
        normal_shuffled_flattened_move_candidates = cp.take_along_axis(
            normal_flattened_move_candidates, normal_shuffled_candidates_idx, axis=1
        )
        normal_shuffled_candidates_max_idx = cp.nanargmax(normal_shuffled_flattened_move_candidates, axis=1)[:, None]

        normal_move_indexes = cp.take_along_axis(
            normal_shuffled_candidates_idx, normal_shuffled_candidates_max_idx, axis=1
        )
        ####

        normal_live_agents_neighbour_flat_positions = live_agents_neighbour_flat_positions[~is_random_move]
        normal_move_new_positions = cp.take_along_axis(
            normal_live_agents_neighbour_flat_positions.reshape(-1, 27), normal_move_indexes,
            axis=1
        ).T[0]
        # (8)
        moving_agents_flat_positions = self.agents_flat_positions[self.agents_state]
        new_agents_flat_positions = moving_agents_flat_positions.copy()

        new_agents_flat_positions[is_random_move] = random_new_positions

        new_agents_flat_positions[~is_random_move] = normal_move_new_positions

        live_agents_indexes = cp.flatnonzero(self.agents_state)

        # (9)
        _, flat_positions_first_entry = cp.unique(
            new_agents_flat_positions, return_index=True
        )

        is_live = cp.zeros_like(new_agents_flat_positions).astype(cp.bool)
        is_live[flat_positions_first_entry] = True

        new_agents_flat_positions[~is_live] = moving_agents_flat_positions[~is_live]
        new_agents_positions = cp.array(cp.unravel_index(new_agents_flat_positions, self.env.shape)).T

        # (10)
        is_live[new_agents_positions[:, 2] == 1] = False

        self._agents_positions[live_agents_indexes] = new_agents_positions
        self.agents_state[live_agents_indexes] = is_live

        self.is_available_env.ravel()[moving_agents_flat_positions] = True
        self.is_available_env.ravel()[new_agents_flat_positions] = False

        self._agents_positions_all_time.append(cp.asnumpy(self._agents_positions))

    @property
    def tracking(self):
        """
        Array with the shape (*, *, 3), which map to these parameters (Agent, Turn, Position).
        It is a whole history for an environment.

        """
        out = np.full((len(self._agents_positions_all_time), *self._agents_positions_all_time[-1].shape,), np.nan)

        for i, pos in enumerate(self._agents_positions_all_time):
            out[i, :pos.shape[0]] = pos - 1

        out = np.moveaxis(out, (0, 1, 2), (1, 0, 2))
        return out

    @property
    def pd_result(self):
        """Generate a dataframe with positions and state of agents."""
        df = pd.DataFrame(cp.asnumpy(self._agents_positions - 1), columns=['x', 'y', 'z'])
        df['is_live'] = cp.asnumpy(self.agents_state)
        return df


class Visualization:
    """Wrapper for different visualization stuff."""

    def __init__(self, env: Environment, accumulation=None, tracked_agents_ratio=1., cpos=(
            (-2170, -1860, 316),
            (142, 145, 670),
            (0.082, 0.09, -1)
        )):
        """
        :param env: Environment instance
        :param tracked_agents_ratio: Ratio of tracked agents.
        :param accumulation: numpy array of accumulations
        :param cpos: camera position
        """
        self.env = env
        self.cpos = cpos
        self.tracked_agents_ratio = tracked_agents_ratio

        if accumulation is not None:
            self.accumulation = np.array(np.unravel_index(np.flatnonzero(accumulation), accumulation.shape)).T
        else:
            self.accumulation = None

    @property
    def _plotter(self):
        """
        Create a plotter with env.
        """
        plotter = pv.Plotter()
        plotter.add_mesh(self.env._raw_env, opacity=0.4, cmap="hot", lighting=True)

        if self.accumulation is not None:
            plotter.add_mesh(self.accumulation, color='green', point_size=3, opacity=0.4, render_points_as_spheres=True)
        return plotter

    def show_animated_movement(self, output_file='move_animation.gif', ):
        """Animate a movement of all agents inside env.

        :param output_file: GIF file, which will be created with animation of movement.
        """
        tracking = self.env.tracking
        size = tracking.shape[0]
        # choices = self.env._rng.choice(
        #     np.arange(size),
        #     int(size * self.tracked_agents_ratio),
        #     replace=False
        # )
        #
        # tracking = tracking[choices]
        tracking = np.moveaxis(tracking, (0, 1, 2), (1, 0, 2))

        plotter = self._plotter
        plotter.show_grid()
        plotter.show_axes_all()

        print('Please wait, points will appear on rendering.')
        tracking[np.isnan(tracking)] = 0

        plotter.show(auto_close=False, interactive_update=True, cpos=self.cpos)
        plotter.open_gif(output_file)

        for i, t in enumerate(tqdm.tqdm(tracking)):
            if i == 0:
                tracking_mesh = pv.PolyData(t)
                plotter.add_mesh(tracking_mesh, color='blue', render_points_as_spheres=True)
            else:
                tracking_mesh.points = t
                plotter.write_frame()

        plotter.close()

    def show_locations(self):
        """
        Show latest locations of agents.
        """
        size = self.env._agents_positions.shape[0]
        choices = self.env._rng.choice(
            np.arange(size),
            int(size * self.tracked_agents_ratio),
            replace=False
        )

        agents = cp.asnumpy(self.env._agents_positions)[choices] - 1

        plotter = self._plotter
        plotter.add_mesh(agents, color='blue', render_points_as_spheres=True)

        plotter.show_grid()
        plotter.show_axes_all()

        plotter.show(cpos=self.cpos)

    def show_tracks(self):
        """
        Show tracks of agents.
        """
        tracking = self.env.tracking
        choices = self.env._rng.choice(
            np.arange(tracking.shape[0]),
            int(tracking.shape[0] * self.tracked_agents_ratio),
            replace=False
        )

        tracking = tracking[choices]
        tracking = np.moveaxis(tracking, (0, 1, 2), (1, 0, 2))

        plotter = self._plotter

        plotter.show_grid()
        plotter.show_axes_all()

        # TODO(!!!!!!): Sadly right now we can't use opacity for every point in array.
        #  (https://github.com/pyvista/pyvista/pull/855) after adding this, we can uncomment this code and it will look
        #  prettier.

        # Calculate opacity for every agent on every turn.
        # First turn for agent will be with min opacity, last - with max.
        # opacity = np.tile(np.arange(tracking.shape[0]), (tracking.shape[1], 1)).T.astype(np.float)
        # opacity_nans = np.min(np.isnan(tracking), axis=2)
        # opacity[opacity_nans] = np.nan
        # opacity_mines, opacity_maxes = np.nanmin(opacity, axis=0), np.nanmax(opacity, axis=0)
        # opacity = (opacity - opacity_mines) / (opacity_maxes + 1 - opacity_mines)
        # opacity[opacity_nans] = 0
        # scalars = np.ones(opacity.shape[1])

        opacity = np.logspace(0.5, 2, num=tracking.shape[0], endpoint=True, base=10.0, dtype=None, axis=0) / 100

        for i, t in enumerate(tqdm.tqdm(tracking)):
            plotter.add_mesh(t, color='blue', opacity=opacity[i], render_points_as_spheres=True)
            # plotter.add_mesh(t, color='blue', scalars=scalars, opacity=opacity[i], render_points_as_spheres=True)

        plotter.show(cpos=self.cpos)


def for_print(arr):
    """ To make an array be more easy to interpret in jupyter output we should change X and Z axes."""
    return np.moveaxis(arr, (-3, -1), (-1, -3))