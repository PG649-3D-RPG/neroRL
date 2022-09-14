import numpy as np
import torch

from neroRL.sampler.buffer import Buffer
from neroRL.utils.worker import Worker

from neroRL.normalization.observation_normalizer import NdNormalizer

import time

class TrajectorySampler():
    """The TrajectorySampler employs n environment workers to sample data for s worker steps regardless if an episode ended.
    Hence, the collected trajectories may contain multiple episodes or incomplete ones."""
    def __init__(self, configs, worker_id, visual_observation_space, vector_observation_space, action_space_shape, n_agents, model, device) -> None:
        """Initializes the TrajectorSampler and launches its environment workers.

        Arguments:
            configs {dict} -- The whole set of configurations (e.g. training and environment configs)
            worker_id {int} -- Specifies the offset for the port to communicate with the environment, which is needed for Unity ML-Agents environments.
            visual_observation_space {box} -- Dimensions of the visual observation space (None if not available)
            vector_observation_space {tuple} -- Dimensions of the vector observation space (None if not available)
            action_space_shape {tuple} -- Dimensions of the action space
            model {nn.Module} -- The model to retrieve the policy and value from
            device {torch.device} -- The device that is used for retrieving the data from the model
        """
        # Set member variables
        self.configs = configs
        self.visual_observation_space = visual_observation_space
        self.vector_observation_space = vector_observation_space
        self.model = model
        self.n_workers = configs["sampler"]["n_workers"]
        self.n_agents = n_agents
        self.worker_steps = configs["sampler"]["worker_steps"]
        self.batch_size = configs["sampler"]["batch_size"]
        self.buffer_size = configs["sampler"]["buffer_size"]
        self.recurrence = None if not "recurrence" in configs["model"] else configs["model"]["recurrence"]
        self.device = device

        self.observationNormalizer = NdNormalizer(vector_observation_space) if configs["model"]["normalize_observations"] else None

        self.action_space_shape = action_space_shape
        self.visual_observation_space = visual_observation_space
        self.vector_observation_space = vector_observation_space
        self.action_space_shape = action_space_shape
        # Create Buffer
        self.buffer = Buffer(self.batch_size, self.buffer_size, self.n_workers, self.n_agents, self.worker_steps, visual_observation_space, vector_observation_space,
                        action_space_shape, self.recurrence, self.device, self.model.share_parameters, self)

        # Launch workers
        self.workers = [Worker(configs["environment"], worker_id + 200 + w) for w in range(self.n_workers)]
        
        # Setup initial observations
        if visual_observation_space is not None:
            self.vis_obs = np.zeros((self.n_workers,) + visual_observation_space.shape, dtype=np.float32)
        else:
            self.vis_obs = None
        if vector_observation_space is not None:
            self.vec_obs = np.zeros((self.n_workers,self.n_agents) + vector_observation_space, dtype=np.float32)
        else:
            self.vec_obs = None

        # Setup initial recurrent cell
        if self.recurrence is not None:
            hxs, cxs = self.model.init_recurrent_cell_states(self.n_workers, self.device)
            if self.recurrence["layer_type"] == "gru":
                self.recurrent_cell = hxs
            elif self.recurrence["layer_type"] == "lstm":
                self.recurrent_cell = (hxs, cxs)
        else:
            self.recurrent_cell = None

        self.agent_id_map = [{} for _ in self.workers]
        self.actions_next_step = [ [] for _ in self.workers]
        # Reset workers
        for worker in self.workers:
            worker.child.send(("reset", None))
        # Grab initial observations
        for w, worker in enumerate(self.workers):
            vis_obs, vec_obs, agent_ids, actions_next_step = worker.child.recv()
            if self.vis_obs is not None:
                self.vis_obs[w] = vis_obs
            if self.vec_obs is not None:
                self.vec_obs[w] = vec_obs

            for agent_id in agent_ids:
                self.agent_id_map[w][agent_id] = len(self.agent_id_map[w].items())

            self.actions_next_step[w] = list(agent_ids[:actions_next_step])

        self.next_step_indices = np.zeros((self.n_workers, self.n_agents), dtype=np.int32)
        self.sampled_steps = 0


    def sample(self, device) -> list:
        """Samples training data (i.e. experience tuples) using n workers for t worker steps.

        Arguments:
            device {torch.device} -- The device that is used for retrieving the data from the model

        Returns:
            {list} -- List of completed episodes. Each episode outputs a dictionary containing at least the
            achieved reward and the episode length.
        """
        episode_infos = []
        self.next_step_indices = np.zeros((self.n_workers, self.n_agents), dtype=np.int32)
        self.sampled_steps = 0

        self.buffer = Buffer(self.batch_size, self.buffer_size, self.n_workers, self.n_agents, self.worker_steps, self.visual_observation_space, self.vector_observation_space,
                        self.action_space_shape, self.recurrence, self.device, self.model.share_parameters, self)


        # Sample actions from the model and collect experiences for training
        for t in range(self.buffer_size):
            # Gradients can be omitted for sampling data
            with torch.no_grad():
                # Save the initial observations and hidden states
                if self.vis_obs is not None:
                    self.buffer.vis_obs[:, t] = torch.tensor(self.vis_obs)
                if self.vec_obs is not None:
                    for w in range(0, self.vec_obs.shape[0]):
                        for a in range(0, self.vec_obs.shape[1]):
                            if np.any(self.vec_obs[w,a,:]): 
                                self.buffer.vec_obs[w,a, self.next_step_indices[w,a]] = torch.tensor(self.vec_obs[w,a,:])

                # Forward the model to retrieve the policy (making decisions), 
                # the states' value of the value function and the recurrent hidden states (if available)
                vis_obs_batch = torch.tensor(self.vis_obs) if self.vis_obs is not None else None
                vec_obs_batch = torch.tensor(self.vec_obs) if self.vec_obs is not None else None
                policy, value, self.recurrent_cell, _ = self.model(vis_obs_batch, vec_obs_batch, self.recurrent_cell)

                for w in range(self.n_workers):
                    for agent_id_with_action in self.actions_next_step[w]:
                        agent_index = self.agent_id_map[w][agent_id_with_action]

                        self.buffer.values[w, agent_index, self.next_step_indices[w, agent_index]] = value[w, agent_index] #TODO apparently numpy cuts one dimension here as it would be (8,1,1) [numpy cuts to (8,)]. Therefore the agent dimension is not accessible for one agent builds

                        self.buffer.std[w, agent_index, self.next_step_indices[w,agent_index]] = policy.stddev[w, agent_index]

                # Sample actions
                action = policy.sample()

                #Might calculate log_props for unused actions
                log_probs = policy.log_prob(action).sum(2) #log probs has shape 8,39 for a 8x10 agent config with axis 1


                pass_actions = [np.zeros((len(self.actions_next_step[x]),)+(self.action_space_shape)) for x in range(self.n_workers)]
 
                for w in range(self.n_workers):
                    for j,a in enumerate(self.actions_next_step[w]):
                        pass_actions[w][j] = action[w, self.agent_id_map[w][a]]
                        
                        self.buffer.actions[w,self.agent_id_map[w][a], self.next_step_indices[w,self.agent_id_map[w][a]]] = action[w,self.agent_id_map[w][a],:]

                        self.buffer.log_probs[w, self.agent_id_map[w][a], self.next_step_indices[w, self.agent_id_map[w][a]]] = log_probs[w, self.agent_id_map[w][a]]
                            
            # Execute actions
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", pass_actions[w]))

            # Retrieve results
            for w, worker in enumerate(self.workers):
                vis_obs, vec_obs, rewards, agent_ids, actions_next_step, episode_end_info = worker.child.recv()

                self.actions_next_step[w] = list(agent_ids[:actions_next_step])

                self.vec_obs = np.zeros_like(self.vec_obs)

                for x in reversed(range(0, len(agent_ids))):
                    agent_id = agent_ids[x]
                    agent_index = self.agent_id_map[w][agent_id]

                    if vec_obs[x].shape is (): # after the initial reset, the get_steps method of the environments return terminal steps with empty observations (shape () ). These are ignored using this if statement
                        continue

                    self.vec_obs[w, agent_index] = self.observationNormalizer.forward(vec_obs[x])
                    self.buffer.rewards[w, agent_index, self.next_step_indices[w, agent_index]] = rewards[x]

                    if x >= actions_next_step:
                        self.buffer.dones[w, agent_index, self.next_step_indices[w, agent_index]] = 1

                    self.next_step_indices[w][agent_id] += 1
                    self.sampled_steps += 1

                    
                if self.vis_obs is not None:
                    self.vis_obs[w] = vis_obs
                if episode_end_info:
                    episode_infos.extend(episode_end_info)
            
            if self.sampled_steps >= self.batch_size:
                break #breaks the loop if enough data has been sampled

        self.buffer.last_filled_indices = self.next_step_indices
        return episode_infos

    def last_vis_obs(self) -> np.ndarray:
        """
        Returns:
            {np.ndarray} -- The last visual observation of the sampling process, which can be used to calculate the advantage.
        """
        return torch.tensor(self.vis_obs) if self.vis_obs is not None else None

    def last_vec_obs(self) -> np.ndarray:
        """
        Returns:
            {np.ndarray} -- The last vector observation of the sampling process, which can be used to calculate the advantage.
        """
        if self.buffer.vec_obs is not None:
            past_vec_obs = np.zeros_like(self.vec_obs)
            for w in range(self.n_workers):
                for a in range(self.n_agents):
                    past_vec_obs[w,a] = self.buffer.vec_obs[w,a, self.next_step_indices[w,a]-1]
            return torch.tensor(past_vec_obs) 
            
        return None
        
    def last_recurrent_cell(self) -> tuple:
        """
        Returns:
            {tuple} -- The latest recurrent cell of the sampling process, which can be used to calculate the advantage.
        """
        return self.recurrent_cell

    def close(self) -> None:
        """Closes the sampler and shuts down its environment workers."""
        try:
            for worker in self.workers:
                worker.child.send(("close", None))
        except:
            pass