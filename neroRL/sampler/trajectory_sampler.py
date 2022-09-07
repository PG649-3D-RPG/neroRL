import numpy as np
import torch

from neroRL.sampler.buffer import Buffer
from neroRL.utils.worker import Worker

from neroRL.normalization.observation_normalizer import NdNormalizer

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

        # Create Buffer
        self.buffer = Buffer(self.batch_size, self.buffer_size, self.n_workers, self.n_agents, self.worker_steps, visual_observation_space, vector_observation_space,
                        action_space_shape, self.recurrence, self.device, self.model.share_parameters, self) #TODO pass n_agents to buffer

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

        self.agent_id_map = [{} for _ in len(self.workers)]
        self.actions_next_step = [ [] for _ in len(self.workers)]
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
                self.agent_id_map[w][agent_id] = len(self.agent_id_map[w].items)

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
                                self.buffer.vec_obs[w,a, self.next_step_indices[w,a]] = self.vec_obs[w,a,:]
                    self.vec_obs = np.zeros_like(self.vec_obs)

                # Forward the model to retrieve the policy (making decisions), 
                # the states' value of the value function and the recurrent hidden states (if available)
                vis_obs_batch = torch.tensor(self.vis_obs) if self.vis_obs is not None else None
                vec_obs_batch = torch.tensor(self.vec_obs) if self.vec_obs is not None else None
                policy, value, self.recurrent_cell, _ = self.model(vis_obs_batch, vec_obs_batch, self.recurrent_cell)

                for w in self.n_workers:
                    for agent_id_with_action in self.actions_next_step[w]:
                        agent_index = self.agent_id_map[w,agent_id_with_action]

                        self.buffer.values[w, agent_index, self.next_step_indices[w, agent_index]] = value[w, agent_index]

                        self.buffer.std[w, agent_index, self.next_step_indices[w,agent_index]] = policy.stddev[w, agent_index]

                # Sample actions
                action = policy.sample()
                #TODO continue from here

                self.buffer.actions[:, t] = action
                
                self.buffer.log_probs[:, t] = policy.log_prob(action).sum(1)
            # Execute actions
            action = self.buffer.actions[:, t].cpu().numpy() # send actions as batch to the CPU, to save IO time
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", action[w]))

            # Retrieve results
            for w, worker in enumerate(self.workers):
                vis_obs, vec_obs, rewards, agent_ids, actions_next_step, episode_end_info = worker.child.recv()

                self.buffer.rewards[w, t] #TODO: adapt all buffers to a [worker, agent, entries] shape, entrys can either be part of the nparray with length worker_steps or a list (depends on what is more efficient)

                for x in range(0, len(agent_ids)):
                    agent_id = agent_ids[x]
                    agent_index = self.agent_id_map[agent_id]

                    self.vec_obs[w, agent_index] = self.observationNormalizer.forward(vec_obs[x])
                    self.buffer.rewards[w, agent_index, self.next_step_indices[w, agent_index]] = rewards[x]

                    if x >= actions_next_step:
                        self.buffer.dones[w, agent_index, self.next_step_indices[w, agent_index]] = 1

                    self.next_step_indices[agent_id] += 1
                    self.sampled_steps += 1

                    
                if self.vis_obs is not None:
                    self.vis_obs[w] = vis_obs
                if episode_end_info:
                    episode_infos.append(episode_end_info)


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
        return torch.tensor(self.vec_obs) if self.vec_obs is not None else None

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