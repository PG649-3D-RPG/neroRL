import torch
import torch.nn as nn

encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first = True)

#(Batch, seq_len, feature_dim)
# feature_dim must be divisible by num_heads
src = torch.rand(32, 10, 512)
out = encoder_layer(src)

print("TransformerEncoderLayer")
print(encoder_layer)
print("src_size: ", src.size())
print("out_size: ", out.size())

# print("src: ", src, sep="\n")
# print("out: ", out, sep="\n")

