import torch
from torch import nn
from neroRL.nn.module import Module

class TransformerEncoder(Module):
    def __init__(self, input_features, num_heads):
        super().__init__()
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=input_features, nhead=num_heads, batch_first = True)
        
    def forward(self, h, sequence_length):
        if sequence_length == 1: 
            h = self.transformer_encoder(h.unsqueeze(1))
                
            h = h.squeeze(1) # Remove sequence length dimension
        else:
            self.buffer.reset_all()
            h_shape = tuple(h.size())
            
            # Reshape the to be fed data to batch_size, sequence_length, data
            h = h.reshape((h_shape[0] // sequence_length), sequence_length, h_shape[1])
            h = self.transformer_encoder(h)
            
            # Reshape to the original tensor size
            h_shape = tuple(h.size())
            h = h.reshape(h_shape[0] * h_shape[1], h_shape[2])
            
        return h