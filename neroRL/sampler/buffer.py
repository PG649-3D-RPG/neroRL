from io import BufferedRandom
import torch
import numpy as np

class Buffer():
    """
    The buffer stores and prepares the training data. It supports recurrent policies.
    """
    def __init__(self, batch_size, buffer_size, num_workers, num_agents, worker_steps, visual_observation_space, vector_observation_space,
                    action_space_shape, recurrence, device, share_parameters, sampler):
        """
        Arguments:
            num_workers {int} -- Number of environments/agents to sample training data
            worker_steps {int} -- Number of steps per environment/agent to sample training data
            num_mini_batches {int} -- Number of mini batches that are used for each training epoch
            visual_observation_space {Box} -- Visual observation if available, else None
            vector_observation_space {tuple} -- Vector observation space if available, else None
            action_space_shape {tuple} -- Shape of the action space
            recurrence {dict} -- None if no recurrent policy is used, otherwise contains relevant details:
                - layer_type {str}, sequence_length {int}, hidden_state_size {int}, hiddens_state_init {str}, reset_hidden_state {bool}
            device {torch.device} -- The device that will be used for training/storing single mini batches
            sampler {TrajectorySampler} -- The current sampler
        """
        self.device = device
        self.sampler = sampler
        self.recurrence = recurrence
        self.sequence_length = recurrence["sequence_length"] if recurrence is not None else None
        self.num_workers = num_workers
        self.worker_steps = worker_steps
        self.num_agents = num_agents
        self.action_space_shape = action_space_shape
        self.vector_observation_space = vector_observation_space
        # self.batch_size = self.num_workers * self.worker_steps
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.rewards = np.zeros((num_workers, num_agents, buffer_size), dtype=np.float32)
        self.actions = torch.zeros((num_workers, num_agents, buffer_size, action_space_shape[0]))
        self.std = torch.zeros((num_workers, num_agents, buffer_size, action_space_shape[0]))
        self.dones = np.zeros((num_workers, num_agents, buffer_size), dtype=np.bool)
        self.last_filled_indices = None
        if visual_observation_space is not None:
            self.vis_obs = torch.zeros((num_workers, num_agents, buffer_size) + visual_observation_space.shape)
        else:
            self.vis_obs = None
        if vector_observation_space is not None:
            self.vec_obs = torch.zeros((num_workers, num_agents, buffer_size,) + vector_observation_space)
        else:
            self.vec_obs = None
        
        if share_parameters:
            self.hxs = torch.zeros((num_workers, worker_steps, recurrence["hidden_state_size"])) if recurrence is not None else None
            self.cxs = torch.zeros((num_workers, worker_steps, recurrence["hidden_state_size"])) if recurrence is not None else None
        else: # if parameters are not shared then add two extra dimensions for adding enough capacity to store the hidden states of the actor and critic model
            self.hxs = torch.zeros((num_workers, worker_steps, recurrence["hidden_state_size"], 2)) if recurrence is not None else None
            self.cxs = torch.zeros((num_workers, worker_steps, recurrence["hidden_state_size"], 2)) if recurrence is not None else None

        self.log_probs = torch.zeros((num_workers, num_agents, buffer_size))
        self.values = torch.zeros((num_workers, num_agents, buffer_size))
        self.advantages = torch.zeros((num_workers, num_agents, buffer_size))
        self.num_sequences = 0
        self.actual_sequence_length = 0

    def calc_advantages(self, last_value, gamma, lamda):
        """Generalized advantage estimation (GAE)

        Arguments:
            last_value {numpy.ndarray} -- Value of the last agent's state
            gamma {float} -- Discount factor
            lamda {float} -- GAE regularization parameter
        """
        with torch.no_grad():
            last_advantage = 0
            mask = torch.tensor(self.dones).logical_not() # mask values on terminal states
            rewards = torch.tensor(self.rewards)

            for t in reversed(range(self.buffer_size)):
                last_value = last_value * mask[:, :, t]
                last_advantage = last_advantage * mask[:, :, t]
                delta = rewards[:, t] + gamma * last_value - self.values[:, :, t]
                last_advantage = delta + gamma * lamda * last_advantage
                self.advantages[: , :, t] = last_advantage
                last_value = self.values[:, :, t]


    def prepare_batch_dict(self):
        """
        Flattens the training samples and stores them inside a dictionary.
        If a recurrent policy is used, the data is split into episodes or sequences beforehand.
        """
        # Supply training samples
        samples = {
            "actions": torch.zeros((self.batch_size,self.action_space_shape)),
            "values": torch.zeros((self.batch_size,)),
            "log_probs": torch.zeros((self.batch_size,)),
            "advantages": torch.zeros((self.batch_size,)),
        }

    	# Add available observations to the dictionary
        if self.vis_obs is not None:
            samples["vis_obs"] = self.vis_obs
        if self.vec_obs is not None:
            samples["vec_obs"] = torch.zeros((self.batch_size,self.vector_observation_space))
      
        filled_indices = 0
        for w in range(self.num_workers):
            for a in range(self.num_agents):
                if filled_indices + self.last_filled_indices[w,a] >= self.batch_size:
                    self.last_filled_indices[w,a] = self.batch_size - filled_indices
                samples["actions"][filled_indices : filled_indices + self.last_filled_indices[w,a]] = self.actions[w,a, : self.last_filled_indices[w,a]]
                samples["values"][filled_indices : self.last_filled_indices[w,a]] = self.values[w,a, : self.last_filled_indices[w,a]]
                samples["log_probs"][filled_indices : self.last_filled_indices[w,a]] = self.log_probs[w,a, : self.last_filled_indices[w,a]]
                samples["advantages"][filled_indices : self.last_filled_indices[w,a]] = self.advantages[w,a, : self.last_filled_indices[w,a]]
                samples["vec_obs"][filled_indices : self.last_filled_indices[w,a]] = self.vec_obs[w,a, : self.last_filled_indices[w,a]]
                filled_indices += self.last_filled_indices[w,a]

    def mini_batch_generator(self, num_mini_batches):
        """A generator that returns a dictionary containing the data of a whole minibatch.
        This mini batch is completely shuffled.

        Arguments:
            num_mini_batches {int} -- Number of the to be sampled mini batches

        Yields:
            {dict} -- Mini batch data for training
        """
        # Prepare indices (shuffle)
        indices = torch.randperm(self.batch_size)
        mini_batch_size = self.batch_size // num_mini_batches
        for start in range(0, self.batch_size, mini_batch_size):
            # Compose mini batches
            end = start + mini_batch_size
            mini_batch_indices = indices[start: end]
            mini_batch = {}
            for key, value in self.samples_flat.items():
                mini_batch[key] = value[mini_batch_indices].to(self.device)
            yield mini_batch

    def refresh(self, model, gamma, lamda):
        """Refreshes the buffer with the current model.

        Arguments:
            model {nn.Module} -- The model to retrieve the policy and value from
            gamma {float} -- Discount factor
            lamda {float} -- GAE regularization parameter
        """
        # Refresh advantages
        _, last_value, _, _ = model(self.sampler.last_vis_obs(), self.sampler.last_vec_obs(), self.sampler.last_recurrent_cell())
        self.calc_advantages(last_value, gamma, lamda)
        
        # Refresh batches
        self.prepare_batch_dict() 