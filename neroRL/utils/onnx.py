import torch
from typing import Tuple
import numpy as np
from neroRL.nn.module import Module

class TensorNames:
    batch_size_placeholder = "batch_size"
    sequence_length_placeholder = "sequence_length"
    vector_observation_placeholder = "vector_observation"
    recurrent_in_placeholder = "recurrent_in"
    visual_observation_placeholder_prefix = "visual_observation_"
    observation_placeholder_prefix = "obs_"
    previous_action_placeholder = "prev_action"
    action_mask_placeholder = "action_masks"
    random_normal_epsilon_placeholder = "epsilon"

    value_estimate_output = "value_estimate"
    recurrent_output = "recurrent_out"
    memory_size = "memory_size"
    version_number = "version_number"

    continuous_action_output_shape = "continuous_action_output_shape"
    discrete_action_output_shape = "discrete_action_output_shape"
    continuous_action_output = "continuous_actions"
    discrete_action_output = "discrete_actions"
    deterministic_continuous_action_output = "deterministic_continuous_actions"
    deterministic_discrete_action_output = "deterministic_discrete_actions"

    # Deprecated TensorNames entries for backward compatibility
    is_continuous_control_deprecated = "is_continuous_control"
    action_output_deprecated = "action"
    action_output_shape_deprecated = "action_output_shape"

    @staticmethod
    def get_visual_observation_name(index: int) -> str:
        """
        Returns the name of the visual observation with a given index
        """
        return TensorNames.visual_observation_placeholder_prefix + str(index)

    @staticmethod
    def get_observation_name(index: int) -> str:
        """
        Returns the name of the observation with a given index
        """
        return TensorNames.observation_placeholder_prefix + str(index)

class OnnxExporter: #modelled after mlagents/trainers/torch/model_serialization.py
    def __init__(self, actor, observation_space):
        self.actor = actor
         
        dummy_obs = torch.zeros(
                [1] + list(OnnxExporter._get_onnx_shape(observation_space))
            )

        self.dummy_input = (dummy_obs)

        self.input_names = ["VectorInput"]

        self.output_names = ['Actions']

        self.dynamic_axes = {name: {0: "batch"} for name in self.input_names}

        # self.output_names = [TensorNames.version_number, TensorNames.memory_size]
        # self.output_names += [
        #     TensorNames.continuous_action_output,
        #     TensorNames.continuous_action_output_shape,
        #     TensorNames.deterministic_continuous_action_output,
        # ]
        self.dynamic_axes.update(
            {TensorNames.continuous_action_output: {0: "batch"}}
        )


    @staticmethod
    def _get_onnx_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Converts the shape of an observation to be compatible with the NCHW format
        of ONNX
        """
        if len(shape) == 3:
            return shape[2], shape[0], shape[1]
        return shape

    def export_onnx(self, path):
        print("Exporting onnx")
        torch.onnx.export(
            self.actor,
            self.dummy_input,
            path,
            opset_version=12,
            input_names=self.input_names,
            output_names=self.output_names,
            #dynamic_axes=self.dynamic_axes
        )
        print("Exported to onnx")


def onnx_export(model, input_size, path):
    model.eval()
    dummy_input = torch.randn(1, input_size, requires_grad=True) 
    print("Trying to export onnx")
    torch.onnx.export(
        model,
        dummy_input,
        path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names = ['VectorInput'],
        output_names = ['Actions'],
        verbose = True
    )
    print("Finished exporting onnx")


class ActorExporter(Module):

    def __init__(self, vec_encoder, body, head, normalizer):
        super().__init__()
        self.vec_encoder = vec_encoder
        self.body = body
        self.head = head
        self.normalization_mean = torch.tensor(normalizer.running_mean)
        self.normalization_variance = torch.tensor(normalizer.running_variance)
        self.normalization_steps = torch.tensor(normalizer.normalization_steps)

    def forward(self, vec_obs):

        normalized_state = torch.clip(
            (vec_obs - self.normalization_mean)
            / np.sqrt(self.normalization_variance / self.normalization_steps),
            -5,
            5,
        )

        
        #h1 = self.vec_encoder(torch.tensor(normalized_state).float())
        h1 = self.vec_encoder(normalized_state.float())
        h2 = self.body(h1)
        policy = self.head(h2)
        return policy.sample()

