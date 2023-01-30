import torch 
from torch import nn
import torch.nn.functional as F
import torch.nn as nn

class RecurrentNetwork(nn.Module):
    """
    Recurrent network for neurogym experiments. Stores activations for easy extraction.
    Uses leaky relu activation for continuous measures
    """
    def __init__(self, input_size, output_size, args):
        super(RecurrentNetwork, self).__init__()
        self.hidden_size = args.hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = args.batch_size
        self.i2h = nn.Linear(input_size + args.hidden_size, args.hidden_size)
        self.i2o = nn.Linear(input_size + args.hidden_size, output_size)
        self.activations = []
        self.h_activations = []
        self.o_activations = []

    def forward(self, x): 
        self.activations = []
        self.h_activations = []
        self.o_activations = []
        h = self.init_hidden()
        x = torch.tensor(x, dtype = torch.float)
        for t in range(0, x.shape[0]): 
            x_t = x[t].reshape(1, self.batch_size, self.input_size)
            xh = torch.cat((x_t, h), 2)
            h = F.leaky_relu(self.i2h(xh)) 
            out_t = F.leaky_relu(self.i2o(xh))
            self.h_activations.append(h.detach().cpu().numpy())
            self.activations.append(out_t.detach().cpu().numpy())
            self.o_activations.append(out_t.detach().cpu().numpy())
            if t == 0:
                out = out_t
            else:
                out = torch.cat((out, out_t), 0)
        return out

    def init_hidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_size)