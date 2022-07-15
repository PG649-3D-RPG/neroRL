from torch import torch, nn
import numpy as np


class TensorNormalizer(nn.Module):
    def __init__(self, vec_obs_size: int):
        super().__init__()
        self.register_buffer("normalization_steps", torch.tensor(1))
        self.register_buffer("running_mean", torch.zeros(vec_obs_size))
        self.register_buffer("running_variance", torch.ones(vec_obs_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.update(inputs)
        normalized_state = torch.clamp(
            (inputs - self.running_mean)
            / torch.sqrt(self.running_variance / self.normalization_steps),
            -5,
            5,
        )
        return normalized_state

    def update(self, vector_input: torch.Tensor) -> None:
        with torch.no_grad():
            steps_increment = vector_input.size()[0]
            total_new_steps = self.normalization_steps + steps_increment

            input_to_old_mean = vector_input - self.running_mean
            new_mean: torch.Tensor = self.running_mean + (
                input_to_old_mean / total_new_steps
            ).sum(0)

            input_to_new_mean = vector_input - new_mean
            new_variance = self.running_variance + (
                input_to_new_mean * input_to_old_mean
            ).sum(0)
            # Update references. This is much faster than in-place data update.
            self.running_mean: torch.Tensor = new_mean
            self.running_variance: torch.Tensor = new_variance
            self.normalization_steps: torch.Tensor = total_new_steps


class NdNormalizer:
    def __init__(self, vec_obs_size: int):
        super().__init__()
        self.normalization_steps = np.array(1)
        self.running_mean = np.zeros(vec_obs_size)
        self.running_variance = np.ones(vec_obs_size)

    def forward(self, inputs):
        self.update(inputs)
        normalized_state = np.clip(
            (inputs - self.running_mean)
            / np.sqrt(self.running_variance / self.normalization_steps),
            -5,
            5,
        )
        return normalized_state

    def update(self, vector_input):
        steps_increment = vector_input.shape[0]
        total_new_steps = self.normalization_steps + steps_increment

        input_to_old_mean = vector_input - self.running_mean
        new_mean = self.running_mean + (
            input_to_old_mean / total_new_steps
        ).sum(0)

        input_to_new_mean = vector_input - new_mean
        new_variance = self.running_variance + (
            input_to_new_mean * input_to_old_mean
        ).sum(0)

        self.running_mean = new_mean
        self.running_variance = new_variance
        self.normalization_steps = total_new_steps