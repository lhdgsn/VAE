import numpy as np
import torch
from torch import nn

class Encoder(nn.Module):
    """
    p(z|x)
    """
    def __init__(self, n_features, z_dim, layer_sizes):
        """
        n_features (int): number of input features per observation
        z_dim (int): number of latent variables
        layer_size (list): number of hidden units for each encoder layer
        """
        self.n_layers = len(layer_sizes)

        self.model_layers = nn.ModuleList()

        # add input layer
        self.model_layers.append(nn.Linear(n_features, layer_sizes[0]))
        self.model_layers.append(nn.ReLU())
        
        if(self.n_layers>2):
            for i_layer in range(self.n_layers-1):
                self.model_layers.append(nn.Linear(layer_sizes[i_layer], layer_sizes[i_layer+1]))
                self.model_layers.append(nn.ReLU())
        
        # output layers for means and variances of latent variables
        self.output_means = nn.Linear(layer_sizes[-1], z_dim)
        self.output_vars = nn.Linear(layer_sizes[-1], z_dim)

    def forward(self, x):
        for model_layer in self.model_layers:
            x = model_layer(x)

        means = self.output_means(x)
        vars = self.output_vars(x)

        return (means, vars)

class Decoder(nn.Module):
    """
    Implements the generative model (decoder):
    Standard: p(x|z)
    Conditional: p(x|z,c)
    Auto-regressive: p(x_i|z,x_{j<i})
    """
    def __init__(self, z_dim, n_features, layer_sizes, is_autoregressive):
        """
        n_features (int): number of input features per observation
        z_dim (int): number of latent variables
        layer_size (list): number of hidden units for each encoder layer
        is_autoregressive (bool): 
        """
        n_layers = len(layer_sizes)
        self.is_autoregressive = is_autoregressive

        self.model_layers = nn.ModuleList()

        # add input layer
        self.model_layers.append(nn.Linear(z_dim, layer_sizes[0]))
        self.model_layers.append(nn.ReLU())
        
        if(n_layers>2):
            for i_layer in range(n_layers-1):
                self.model_layers.append(nn.Linear(layer_sizes[i_layer], layer_sizes[i_layer+1]))
                self.model_layers.append(nn.ReLU())
        
        # output layer
        self.output_layer = nn.Linear(layer_sizes[-1], n_features)

    def autoregressive_mask(self):
        pass

    def forward(self):
        for model_layer in self.model_layers:
            x = model_layer(x)
        
        if(self.is_autoregressive):
            pass

        return self.output_layer(x)

class VAE(nn.Module):
    def __init__(self, n_features, z_dim, layer_sizes, activation=None, n_batches=1, is_autoregressive=False, is_conditional=False, seed=0):
        """
        """
        self.input_dim = n_features
        self.z_dim = z_dim
        self.is_autoregressive = is_autoregressive
        self.is_conditional = is_conditional

        self.rng = np.random.default_rng(seed)

        # define encoder network
        self.encoder = Encoder(n_features, z_dim, layer_sizes)

        # define decoder network
        self.decoder = Decoder(n_features, z_dim, layer_sizes, is_autoregressive)

        # latent batch offset vector
        if(n_batches>1):
            self.batch_offset = nn.parameter.Parameter(torch.rand(z_dim, n_batches))
        else:
            self.batch_offset = torch.zeros(z_dim, n_batches) # no batch correction

    def elbo_loss(self, x, x_out):
        """
        """
        pass

    def forward(self, x, condition_labels=None):
        """
        """
        # append condition labels if relevant (may be continuous)
        if(self.is_conditional):
            x = torch.concat((x, condition_labels), dim=1)

        # multivariate Gaussian prior
        # generate means and variances for latent variables z
        (means, vars) = self.encoder(x)
        z = means + vars * self.rng.normal(0, 1, size=x.shape[0]) + self.batch_offset

        # append condition labels if relevant (may be continuous)
        if(self.is_conditional):
            z = torch.concat((z, condition_labels), dim=1)

        x_out = self.decoder(z)

        return x_out

