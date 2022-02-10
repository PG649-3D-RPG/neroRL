import numpy as np

from gym import error, spaces
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.environment_parameters_channel import (EnvironmentParametersChannel,)
from mlagents_envs.side_channel.engine_configuration_channel import (EngineConfigurationChannel,)
from neroRL.environments.env import Env
from random import randint

class MultiUnityWrapper(Env):
    """This class wraps Unity environments that contain multiple environment instances.

    This wrapper has notable constraints:
        - Only one visual observation
        - Only discrete and multi-discrete action spaces (no continuous action space)"""

    def __init__(self, env_path, reset_params, worker_id = 1, no_graphis = False, realtime_mode = False):
        """Instantiates the Unity Environment from a specified executable.
        
        Arguments:
            env_path {string} -- Path to the executable of the environment
            reset_params {dict} -- Reset parameters of the environment such as the seed
        
        Keyword Arguments:
            worker_id {int} -- Port of the environment"s instance (default: {1})
            no_graphis {bool} -- Whether to allow the executable to render or not (default: {False})
            realtime_mode {bool} -- Whether to run the environment in real time or as fast as possible (default: {False})
        """
        # Initialize channels
        self.reset_parameters = EnvironmentParametersChannel()
        self.engine_config = EngineConfigurationChannel()

        # Prepare default reset parameters
        self._default_reset_parameters = {}
        for key, value in reset_params.items():
            self._default_reset_parameters[key] = value
            if key != "start-seed" or key != "num-seeds":
                self.reset_parameters.set_float_parameter(key, value)

        self._realtime_mode = realtime_mode
        if realtime_mode:
            self.engine_config.set_configuration_parameters(time_scale=1.0, width=1280, height=720)
        else:
            self.engine_config.set_configuration_parameters(time_scale=30.0, width=256, height=256)

        # Launch the environment's executable
        self._env = UnityEnvironment(file_name = env_path, worker_id = worker_id, no_graphics = no_graphis, side_channels=[self.reset_parameters, self.engine_config], timeout_wait=300)
        # If the Unity Editor chould be used instead of a build
        # self._env = UnityEnvironment(file_name = None, worker_id = 0, no_graphics = no_graphis, side_channels=[self.reset_parameters, self.engine_config])

        # Reset the environment
        self._env.reset()
        # Retrieve behavior configuration
        self._behavior_name = list(self._env.behavior_specs)[0]
        self._behavior_spec = self._env.behavior_specs[self._behavior_name]

        # Check whether this Unity environment is supported
        self._verify_environment()

        # Aquire agent count
        info, _ = self._env.get_steps(self._behavior_name)
        self._agent_count = len(info)

        # Set action space properties
        if self._behavior_spec.action_spec.is_discrete():
            num_action_branches = self._behavior_spec.action_spec.discrete_size
            action_branch_dimensions = self._behavior_spec.action_spec.discrete_branches
            if num_action_branches == 1:
                self._action_space = spaces.Discrete(action_branch_dimensions[0])
            else:
                self._action_space = spaces.MultiDiscrete(action_branch_dimensions)

        # Count visual and vector observations
        self._num_vis_obs, self._num_vec_obs = 0, 0
        self._vec_obs_indices = []
        for index, obs in enumerate(self._behavior_spec.observation_specs):
            if len(obs[0]) > 1:
                self._num_vis_obs = self._num_vis_obs + 1
                self._vis_obs_index = index
            else:
                self._num_vec_obs = self._num_vec_obs + 1
                self._vec_obs_indices.append(index)

        # Set visual observation space property
        if self._num_vis_obs == 1:
            vis_obs_shape = self._behavior_spec.observation_specs[self._vis_obs_index].shape

            self._visual_observation_space = spaces.Box(
                low = 0,
                high = 1.0,
                shape = vis_obs_shape,
                dtype = np.float32)
        else:
            self._visual_observation_space = None

        # Set vector observation space property
        if self._num_vec_obs > 0:
            # Determine the length of vec obs by summing the length of each distinct one
            vec_obs_length = sum([self._behavior_spec.observation_specs[i][0][0] for i in self._vec_obs_indices])
            self._vector_observatoin_space = (vec_obs_length, )
        else:
            self._vector_observatoin_space = None

    @property
    def unwrapped(self):
        """        
        Returns:
            {UnityWrapper} -- Environment in its vanilla (i.e. unwrapped) state
        """
        return self
    
    @property
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        return self._action_space

    @property
    def action_names(self):
        return None

    @property
    def get_episode_trajectory(self):
        """Returns the trajectory of an entire episode as dictionary (vis_obs, vec_obs, rewards, actions). 
        """
        return Exception("This method should not be called on this particular wrapper.")

    @property
    def visual_observation_space(self):
        return self._visual_observation_space

    @property
    def vector_observation_space(self):
        return self._vector_observatoin_space

    def reset(self, reset_params = None):
        """Resets the environment based on a global or just specified config.
        
        Keyword Arguments:
            config {dict} -- Reset parameters to configure the environment (default: {None})
        
        Returns:
            {numpy.ndarray} -- Visual observation
            {numpy.ndarray} -- Vector observation
        """
        # Track rewards of an entire episode
        self._rewards = []

        # Use initial or new reset parameters
        if reset_params is None:
            reset_params = self._default_reset_parameters
        else:
            reset_params = reset_params

        # Apply reset parameters
        for key, value in reset_params.items():
            # Skip reset parameters that are not used by the Unity environment
            if key != "start-seed" or key != "num-seeds":
                self.reset_parameters.set_float_parameter(key, value)

        # Sample the to be used seed
        if reset_params["start-seed"] > -1:
            seed = randint(reset_params["start-seed"], reset_params["start-seed"] + reset_params["num-seeds"] - 1)
        else:
            # Use unlimited seeds
            seed = -1
        self.reset_parameters.set_float_parameter("seed", seed)

        # Reset and verify the environment
        self._env.reset()
        info, terminal_info = self._env.get_steps(self._behavior_name)
        self._verify_environment()
        
        # Retrieve initial observations
        vis_obs, vec_obs, _, _ = self._process_agent_info(info, terminal_info)

        return vis_obs, vec_obs

    def step(self, actions):
        """Runs one timestep of the environment"s dynamics.
        Once an episode is done, reset() has to be called manually.
                
        Arguments:
            action {List} -- A list of at least one discrete action to be executed by the agent

        Returns:
            {numpy.ndarray} -- Visual observation
            {numpy.ndarray} -- Vector observation
            {float} -- (Total) Scalar reward signaled by the environment
            {bool} -- Whether the episode of the environment terminated
            {dict} -- Further episode information (e.g. cumulated reward) retrieved from the environment once an episode completed
        """
        # Carry out the agent's action
        action_tuple = ActionTuple()
        # A numpy array of shape (num agents, discrete actions) is expected
        action_tuple.add_discrete(actions)
        self._env.set_actions(self._behavior_name, action_tuple)
        self._env.step()
        info, terminal_info = self._env.get_steps(self._behavior_name)

        # Process step results
        vis_obs, vec_obs, reward, done = self._process_agent_info(info, terminal_info)
        self._rewards.append(reward)

        # # Episode information
        # if done:
        #     info = {"reward": sum(self._rewards),
        #             "length": len(self._rewards)}
        # else:
        #     info = None

        return vis_obs, vec_obs, reward, done, info

    def close(self):
        """Shut down the environment."""
        self._env.close()

    def _process_agent_info(self, info, terminal_info):
        """Extracts the observations, rewards, dones, and episode infos.

        Args:
            info {DecisionSteps}: Current state
            terminal_info {TerminalSteps}: Terminal state

        Returns:
            vis_obs {ndarray} -- Visual observation if available, else None
            vec_obs {ndarray} -- Vector observation if available, else None
            reward {float} -- Reward signal from the environment
            done {bool} -- Whether the episode terminated or not
        """
        # Process observations, rewards, dones, ...
        ### In the case of no agent is done
        # if len(terminal_info) == 0:
        #     # Process visual observations
        #     vis_obs = info.obs[self._vis_obs_index] if self.visual_observation_space else None
        #     # Process vector observations
        #     if self.vector_observation_space is not None:
        #         for i, dim in enumerate(self._vec_obs_indices):
        #             if i == 0:
        #                 vec_obs = info.obs[dim]
        #             else:
        #                 vec_obs = np.concatenate((vec_obs, info.obs[dim]))
        #     else:
        #         vec_obs = None
        #     # Process dones
        #     dones = np.zeros(self._agent_count, dtype=bool)
        #     # Process rewards
        #     rewards = info.reward

        # ### In the case of all agents are done
        # if len(info) == 0:
        #     # Process visual observations
        #     vis_obs = terminal_info.obs[self._vis_obs_index] if self.visual_observation_space else None
        #     # Process vector observations
        #     if self.vector_observation_space is not None:
        #         for i, dim in enumerate(self._vec_obs_indices):
        #             if i == 0:
        #                 vec_obs = terminal_info.obs[dim]
        #             else:
        #                 vec_obs = np.concatenate((vec_obs, terminal_info.obs[dim]))
        #     else:
        #         vec_obs = None
        #     # Process dones
        #     dones = np.ones(self._agent_count, dtype=bool)
        #     # Process rewards
        #     rewards = terminal_info.reward

        ### In the case of not all agents are done
        # if len(info) != 0 and len(terminal_info) != 0:
        # Create data placeholders
        vis_obs = np.zeros((self._agent_count, ) + self.visual_observation_space.shape, dtype=np.float32) if self.visual_observation_space else None
        vec_obs = np.zeros((self._agent_count, ) + self.vector_observation_space.shape, dtype=np.float32) if self.vector_observation_space else None
        rewards = np.zeros(self._agent_count, dtype=np.float32)
        dones = np.zeros(self._agent_count, dtype=bool)
        for i in range(len(info)):
            if vis_obs is not None:
                vis_obs[info.agent_id[i]] = info.obs[self._vis_obs_index][i]
            if vec_obs is not None:
                for i, dim in enumerate(self._vec_obs_indices):
                    if i == 0:
                        vec_obs[info.agent_id[i]] = info.obs[dim]
                    else:
                        vec_obs[info.agent_id[i]] = np.concatenate((vec_obs, info.obs[dim][i]))
            rewards[info.agent_id[i]] = info.reward[i]
        for i in range(len(terminal_info)):
            if vis_obs is not None:
                vis_obs[terminal_info.agent_id[i]] = terminal_info.obs[self._vis_obs_index][i]
            if vec_obs is not None:
                for i, dim in enumerate(self._vec_obs_indices):
                    if i == 0:
                        vec_obs[terminal_info.agent_id[i]] = terminal_info.obs[dim][i]
                    else:
                        vec_obs[terminal_info.agent_id[i]] = np.concatenate((vec_obs, terminal_info.obs[dim][i]))
            rewards[terminal_info.agent_id[i]] = terminal_info.reward[i]
            dones[terminal_info.agent_id[i]] = True

        return vis_obs, vec_obs, rewards, dones

    def _verify_environment(self):
        # Verify number of agent behavior types
        if len(self._env.behavior_specs) != 1:
            raise UnityEnvironmentException("The unity environment containts more than one agent type.")
        # Verify action space type
        if not self._behavior_spec.action_spec.is_discrete() or self._behavior_spec.action_spec.is_continuous():
            raise UnityEnvironmentException("Continuous action spaces are not supported. " 
                                            "Only discrete and MultiDiscrete spaces are supported.")
        # Verify that at least one observation is provided
        num_vis_obs = 0
        num_vec_obs = 0
        for obs_spec in self._behavior_spec.observation_specs:
            if len(obs_spec.shape) == 3:
                num_vis_obs += 1
            elif(len(obs_spec.shape)) == 1:
                num_vec_obs += 1
        if num_vis_obs == 0 and num_vec_obs == 0:
            raise UnityEnvironmentException("The unity environment does not contain any observations.")
        # Verify number of visual observations
        if num_vis_obs > 1:
            raise UnityEnvironmentException("The unity environment contains more than one visual observation.")
        
class UnityEnvironmentException(error.Error):
    """Any error related to running the Unity environment."""
    pass