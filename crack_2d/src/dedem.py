import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    
    def __init__(self, size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(size, size), nn.Tanh(),
            nn.Linear(size, size), nn.Tanh()
        )
    def forward(self, x):
        return x + self.layers(x)

class DEDEM_Net(nn.Module):
    
    def __init__(self, input_size=3, hidden_size=30, output_size=1, num_blocks=2):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.Tanh()]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(hidden_size))
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
    

# class ResidualBlock(nn.Module):
#     def __init__(self, size, dropout_rate=0.0):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(size, size), nn.Tanh(),
#             nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
#             nn.Linear(size, size), nn.Tanh()
#         )
#         self.dropout_rate = dropout_rate

#         # Initialize weights properly for tanh activation
#         self._initialize_weights()
#     def _initialize_weights(self):
#         """Xavier/Glorot initialization cho tanh activation"""
#         for module in self.layers.modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight)
#                 nn.init.zeros_(module.bias)
#     def forward(self, x):
#         return x + self.layers(x)

# class DEDEM_Net(nn.Module):
#     def __init__(self, input_size=3, hidden_size=30, output_size=1, num_blocks=2, dropout_rate=0.0, use_layer_norm=False):
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.num_blocks = num_blocks

#         # Input layer
#         self.input_layer = nn.Linear(input_size, hidden_size)
#         self.input_activation = nn.Tanh()
#         self.layer_norm = nn.LayerNorm(hidden_size) if use_layer_norm else None

#         # Residual blocks
#         self.residual_blocks = nn.ModuleList([
#             ResidualBlock(hidden_size, dropout_rate)
#             for _ in range(num_blocks)
#         ])

#         # Output layer
#         self.output_layer = nn.Linear(hidden_size, output_size)

#         # Initialize weights
#         self._initialize_weights()

#     def _initialize_weights(self):
#         """
#         Weight initialization cho DEDEM.

#         Sử dụng Xavier initialization cho tanh activation functions.
#         """
#         nn.init.xavier_uniform_(self.input_layer.weight)
#         nn.init.zeros_(self.input_layer.bias)

#         nn.init.xavier_uniform_(self.output_layer.weight)
#         nn.init.zeros_(self.output_layer.bias)

#     def forward(self, x):
#         out = self.input_activation(self.input_layer(x))

#         # Layer normalization nếu có
#         if self.layer_norm is not None:
#             out = self.layer_norm(out)

#         # Residual blocks
#         for block in self.residual_blocks:
#             out = block(out)

#         # Output layer (không có activation cuối)
#         out = self.output_layer(out)

#         return out