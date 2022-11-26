import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from neroRL.distributions.distributions import TanhGaussianDistInstance
import torch.nn.functional as F

from neroRL.nn.module import Module

class ContinuousActionPolicy(Module):
    def __init__(self, in_features, pre_head_features, action_space_shape, activ_fn, tanh_squashing):
        super().__init__()
        # Set the activation function
        self.activ_fn = activ_fn
        # Linear layer before head
        self.linear = nn.Linear(in_features=in_features, out_features=pre_head_features)
        # nn.init.orthogonal_(self.linear.weight, np.sqrt(2))
        # nn.init.orthogonal_(self.linear.weight, 1)
        nn.init.kaiming_normal_(self.linear.weight.data, nonlinearity="linear")
        torch.zero_(self.linear.bias.data)

        # Mean of the normal distribution
        self.mu = nn.Linear(in_features=pre_head_features, out_features=action_space_shape[0])
        # nn.init.orthogonal_(self.mu.weight, np.sqrt(0.01))
        # nn.init.orthogonal_(self.mu.weight, 0.01)
        # nn.init.constant_(self.mu.bias, 0.0)
        nn.init.kaiming_normal_(self.mu.weight.data, nonlinearity="linear")
        self.mu.weight.data *= 0.2 #kernel gain taken from MLAgents from distributions.py for GaussianDistribution
        torch.zero_(self.mu.bias.data)

        self.tanh_squashing = tanh_squashing

        # Std of the normal distribution as a learnable parameter
        self.logstd = nn.Parameter(torch.zeros(1, np.prod(action_space_shape), requires_grad=True))

        #uncomment this line if a fixed std should be used
        # self.logstd = torch.full((1, np.prod(action_space_shape)), 0.2, dtype=torch.float32)

    def forward(self, h):
        # Feed hidden layer
        h = self.activ_fn(self.linear(h))

        mu = self.mu(h)
        logstd = self.logstd.expand_as(mu)
        std = torch.exp(logstd)

        if(self.tanh_squashing):
            return TanhGaussianDistInstance(mu, std)
        else:
            return Normal(mu, std)

class ValueEstimator(Module):
    """Estimation of the value function as part of the agnet's critic"""
    def __init__(self, in_features, pre_head_features, activ_fn):
        """
        Arguments:
            in_features {int} -- Number of to be fed features
            activ_fn {function} -- The to be applied activation function to the linear layer
        """
        super().__init__()
        # Set the activation function
        self.activ_fn = activ_fn
        # Linear layer before head
        self.linear = nn.Linear(in_features=in_features, out_features=pre_head_features)
        nn.init.orthogonal_(self.linear.weight, np.sqrt(2))
        # Value head
        self.value = nn.Linear(in_features=pre_head_features, out_features=1)
        # nn.init.orthogonal_(self.value.weight, 1)
        nn.init.xavier_uniform_(self.value.weight.data)

        print("value head hidden size ", str(pre_head_features))

    def forward(self, h):
        """
        Arguments:
            h {toch.tensor} -- The fed input data

        Returns:
            {torch.tensor} -- Estimated value
        """
        h = self.activ_fn(self.linear(h))
        value = self.value(h)
        return value.squeeze(len(value.shape)-1)#.reshape(-1) #TODO fix this reshape 

class AdvantageEstimator(Module):
    """Used by the DAAC Algorithm by Raileanu & Fergus, 2021, https://arxiv.org/abs/2102.10330"""
    def __init__(self, in_features, action_space_shape):
        """
        Arguments:
            in_features {int} -- Number of to be fed features
            action_space_shape {tuple} -- Dimensions of the action space
        """
        super().__init__()
        # Set action space
        self.action_space_shape = action_space_shape
        # Calculate the total number of actions
        self.total_num_actions = sum(action_space_shape)
        # Advantage head
        self.advantage = nn.Linear(in_features=in_features + self.total_num_actions, out_features=1)
        nn.init.orthogonal_(self.advantage.weight, 0.01)

    def forward(self, h, actions):
        """
        Arguments:
            h {toch.tensor} -- The fed input data
            actions {toch.tensor} -- The actions of the agent
            device {torch.device} -- Current device

        Returns:
            {torch.tensor} -- Estimated advantage function
        """
        if actions is None:
            one_hot_actions = torch.zeros(h.shape[0], self.total_num_actions).to(next(self.parameters()).device)
            h = torch.cat((h, one_hot_actions), dim=1)
        else:
            for i in range(len(self.action_space_shape)):
                action, num_actions = actions[:, i], self.action_space_shape[i]
                one_hot_actions = F.one_hot(action.squeeze(-1), num_actions).float()
                h = torch.cat((h, one_hot_actions), dim=1)
        
        return self.advantage(h).reshape(-1)
