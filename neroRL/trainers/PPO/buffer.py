import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class Buffer():
    """The buffer stores and prepares the training data. It supports recurrent policies.
    """
    def __init__(self, n_workers, worker_steps, n_mini_batch, visual_observation_space, vector_observation_space, action_space_shape, use_recurrent, hidden_state_size, sequence_length, device, mini_batch_device):
        """
        Arguments:
            n_workers {int} -- Number of environments/agents to sample training data
            worker_steps {int} -- Number of steps per environment/agent to sample training data
            n_mini_batch {int} -- Number of mini batches that are used for each training epoch
            visual_observation_space {Box} -- Visual observation if available, else None
            vector_observation_space {tuple} -- Vector observation space if available, else None
            action_space_shape {tuple} -- Shape of the action space
            use_recurrent {bool} -- Whether to use a recurrent model
            hidden_state_size {int} -- Size of the GRU layer (short-term memory)
            device {torch.device} -- The device that will be used for training/storing single mini batches
            mini_batch_device {torch.device} -- The device that will be used for storing the whole batch of data. This should be CPU if not enough VRAM is available.
        """
        self.device = device
        self.use_recurrent = use_recurrent
        self.sequence_length = sequence_length
        self.n_workers = n_workers
        self.worker_steps = worker_steps
        self.n_mini_batch = n_mini_batch
        self.batch_size = self.n_workers * self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batch
        self.mini_batch_device = mini_batch_device
        self.rewards = np.zeros((n_workers, worker_steps), dtype=np.float32)
        self.actions = np.zeros((n_workers, worker_steps, len(action_space_shape)), dtype=np.int32)
        self.dones = np.zeros((n_workers, worker_steps), dtype=np.bool)
        if visual_observation_space is not None:
            self.vis_obs = np.zeros((n_workers, worker_steps) + visual_observation_space.shape, dtype=np.float32)
        else:
            self.vis_obs = None
        if vector_observation_space is not None:
            self.vec_obs = np.zeros((n_workers, worker_steps,) + vector_observation_space, dtype=np.float32)
        else:
            self.vec_obs = None
        self.log_probs = np.zeros((n_workers, worker_steps, len(action_space_shape)), dtype=np.float32)
        self.values = np.zeros((n_workers, worker_steps), dtype=np.float32)
        self.advantages = np.zeros((n_workers, worker_steps), dtype=np.float32)

    def calc_advantages(self, last_value, gamma, lamda):
        """Generalized advantage estimation (GAE)

        Arguments:
            last_value {numpy.ndarray} -- Value of the last agent's state
            gamma {float} -- Discount factor
            lamda {float} -- GAE regularization parameter
        """
        last_advantage = 0
        for t in reversed(range(self.worker_steps)):
            mask = 1.0 - self.dones[:, t] # mask value on a terminal state (i.e. done)
            last_value = last_value * mask
            last_advantage = last_advantage * mask
            delta = self.rewards[:, t] + gamma * last_value - self.values[:, t]
            last_advantage = delta + gamma * lamda * last_advantage
            self.advantages[:, t] = last_advantage
            last_value = self.values[:, t]

    def prepare_batch_dict(self, episode_done_indices):
        """Flattens the training samples and stores them inside a dictionary. If a recurrent policy is used, that data is split into episodes or sequences beforehand.
        
        Arguments:
            episode_done_indices {list} -- Nested list that stores the done indices of each worker"""
        # Supply training samples
        samples = {
            'actions': self.actions,
            'values': self.values,
            'log_probs': self.log_probs,
            'advantages': self.advantages
        }

    	# Add observations to dictionary
        if self.vis_obs is not None:
            samples['vis_obs'] = self.vis_obs
        if self.vec_obs is not None:
            samples['vec_obs'] = self.vec_obs

        # If recurrent, split data into episodes and apply zero-padding
        if self.use_recurrent:
            # Append the index of the last element of a trajectory as well, as it "artifically" marks the end of an episode
            for w in range(self.n_workers):
                if len(episode_done_indices[w]) == 0 or episode_done_indices[w][-1] != self.worker_steps - 1:
                    episode_done_indices[w].append(self.worker_steps - 1)
            
            # Split vis_obs, vec_obs, values, advantages, actions and log_probs into episodes and then into sequences
            max_sequence_length = 1
            for key, value in samples.items():
                sequences = []
                for w in range(self.n_workers):
                    start_index = 0
                    for done_index in episode_done_indices[w]:
                        # Split trajectory into episodes
                        episode = value[w, start_index:done_index + 1]
                        start_index = done_index + 1
                        # Split episodes into sequences
                        if self.sequence_length > 0:
                            for start in range(0, len(episode), self.sequence_length):
                                end = start + self.sequence_length
                                sequences.append(episode[start:end])
                                max_sequence_length = self.sequence_length
                        else:
                            # If the sequence length is not set to a proper value, sequences will be based on episodes
                            sequences.append(episode)
                            max_sequence_length = len(episode) if len(episode) > max_sequence_length else max_sequence_length
                
                # Apply zero-padding to ensure that each episode has the same length
                # Therfore we can train batches of episodes in parallel instead of one episode at a time
                for i, sequence in enumerate(sequences):
                    sequences[i] = self.pad_sequence(sequence, max_sequence_length)

                # Stack episodes (target shape: (Episode, Step, Data ...) & apply data to the samples dict
                samples[key] = np.stack(sequences, axis=0)

            # TODO: more intuitive variables...
            self.num_sequences = len(samples["values"])
            self.actual_sequence_length = max_sequence_length
            
        # Flatten all samples
        self.samples_flat = {}
        for key, value in samples.items():
            value = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
            self.samples_flat[key] = torch.tensor(value, dtype = torch.float32, device = self.mini_batch_device)

    def pad_sequence(self, sequence, target_length):
        """Pads a sequence to the target length using zeros.

        Args:
            sequence {numpy.ndarray}: The to be padded array (i.e. sequence)
            target_length {int}: The desired length of the sequence

        Returns:
            {numpy.ndarray}: Returns the padded sequence
        """
        # If a tensor is provided, convert it to a numpy array
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.numpy()
        # Determine the number of zeros that have to be added to the sequence
        delta_length = target_length - len(sequence)
        # If the sequence is already as long as the target length, don't pad
        if delta_length <= 0:
            return sequence
        # Construct array of zeros
        if len(sequence.shape) > 1:
            zeros = np.zeros(((delta_length,) + sequence.shape[1:]), dtype=sequence.dtype)
        else:
            zeros = np.zeros(delta_length, dtype=sequence.dtype)
        # Concatenate the zeros to the sequence
        return np.concatenate((sequence, zeros), axis=0)

    def mini_batch_generator(self):
        """A generator that returns a dictionary containing the data of a whole minibatch.
        This mini batch is completely shuffled.

        Yields:
            {dict} -- Mini batch data for training
        """
        # Prepare indices (shuffle)
        indices = torch.randperm(self.batch_size)
        for start in range(0, self.batch_size, self.mini_batch_size):
            # Arrange mini batches
            end = start + self.mini_batch_size
            mini_batch_indices = indices[start: end]
            mini_batch = {}
            for key, value in self.samples_flat.items():
                mini_batch[key] = value[mini_batch_indices].to(self.device)
            yield mini_batch

    def recurrent_mini_batch_generator(self):
        """A recurrent generator that returns a dictionary containing the data of a whole minibatch.
        In comparison to the none-recurrent one, this generator maintains the sequences of the workers' experience trajectories.

        Yields:
            {dict} -- Mini batch data for training
        """
        # Determine the number of episodes per mini batch
        num_eps_per_batch = self.num_sequences // self.n_mini_batch
        num_eps_per_batch = [num_eps_per_batch] * self.n_mini_batch # Arrange a list that determines the episode count for each mini batch
        remainder = self.num_sequences % self.n_mini_batch
        for i in range(remainder):
            num_eps_per_batch[i] += 1 # Add the remainder if the episode count and the number of mini batches do not share a common divider
        # Prepare indices, but only shuffle the episode indices and not the entire batch to ensure that sequences of episodes are maintained
        indices = np.arange(0, self.num_sequences * self.actual_sequence_length).reshape(self.num_sequences, self.actual_sequence_length)
        episode_indices = torch.randperm(self.num_sequences)

        # Compose mini batches
        start = 0
        for num_eps in num_eps_per_batch:
            end = start + num_eps
            mini_batch_indices = indices[episode_indices[start:end]].reshape(-1)
            mini_batch = {}
            for key, value in self.samples_flat.items():
                mini_batch[key] = value[mini_batch_indices].to(self.device)
            start = end
            yield mini_batch
