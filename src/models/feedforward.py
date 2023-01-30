import torch 
from torch import nn
import torch.nn.functional as F
import torch.nn as nn
import copy

class FFNetwork(nn.Module):
    """
    Feedforward network for logic gate experiments. Stores activations for easy extraction.
    """
    def __init__(self, args, dropout, lesion = False):
        super(FFNetwork, self).__init__()
        self.args = args
        self.lesion = lesion
        layer_sizes_copy = copy.deepcopy(args.layer_sizes)
        layer_sizes_copy.insert(0, args.input_size)
        layer_sizes_copy.append(args.output_size) 
        self.fc = nn.ModuleList()
        for i in range(len(layer_sizes_copy)-2):
            self.fc.append(nn.Linear(layer_sizes_copy[i], layer_sizes_copy[i+1], bias = True))
        self.out = nn.Linear(layer_sizes_copy[-2], layer_sizes_copy[-1])
        self.activations = []
        self.drop_layer = nn.Dropout(p=dropout)
        self.mask = []

    def forward(self, x):
        self.activations = []
        x = torch.tensor(x, dtype=torch.float)
        i = 0
        for l in self.fc:
            x = l(x)
            if self.lesion:
                if self.mask is not torch.Tensor:
                    self.mask = torch.FloatTensor(self.mask)
                x = self.mask[i]*x
            x = self.drop_layer(x)
            x = F.relu(x)
            self.activations.append(x.detach().cpu().numpy())
        out = self.out(x)
        self.activations.append(out.detach().cpu().numpy())
        return out