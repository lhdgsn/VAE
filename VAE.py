import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Implements the approximate posterior q(z|x)
    """
    def __init__(self, n_features, z_dim, layer_sizes, n_conditions):
        """
        n_features (int): number of input features per observation
        z_dim (int): number of latent variables
        layer_size (list): number of hidden units for each encoder layer
        """
        super(Encoder, self).__init__()
        self.n_layers = len(layer_sizes)

        self.model_layers = nn.ModuleList()

        # add input layer
        self.model_layers.append(nn.Linear(n_features, layer_sizes[0]))
        self.model_layers.append(nn.ReLU())
        
        if(self.n_layers>1):
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
        logvars = self.output_vars(x)

        return (means, logvars)

class Decoder(nn.Module):
    """
    Implements the generative model (decoder):
    Standard: p(x|z)
    Conditional: p(x|z,c)
    Auto-regressive: p(x_i|z,x_{j<i})
    """
    def __init__(self, z_dim, n_features, layer_sizes, generative_model, is_autoregressive, n_conditions):
        """
        n_features (int): number of input features per observation
        z_dim (int): number of latent variables
        layer_size (list): number of hidden units for each encoder layer
        is_autoregressive (bool): 
        """
        super(Decoder, self).__init__()
        self.n_features = n_features
        layer_sizes.reverse() # for symmetric decoder
        self.is_autoregressive = is_autoregressive
        self.n_conditions = n_conditions

        # add full-dimension autoregressive layer (to be masked to enforce auto-regressive condition)
        if(self.is_autoregressive):
            layer_sizes.append(n_features)

        n_layers = len(layer_sizes)

        self.model_layers = nn.ModuleList()

        # add input layer
        self.model_layers.append(nn.Linear(z_dim, layer_sizes[0]))
        self.model_layers.append(nn.ReLU())
        
        # add hidden layers
        if(n_layers>1):
            for i_layer in range(n_layers-1):
                self.model_layers.append(nn.Linear(layer_sizes[i_layer], layer_sizes[i_layer+1]))
                self.model_layers.append(nn.ReLU())
        
        # output layers (depends on generative model)
        self.output_layers = nn.ModuleDict()
        self.output_activations = nn.ModuleDict()
        
        if(generative_model == 'gaussian'):
            self.output_layers['mean'] = nn.Linear(layer_sizes[-1], n_features)
            self.output_layers['sigma'] = nn.Linear(layer_sizes[-1], n_features)
        
        elif(generative_model == 'nb'):
            self.output_layers['n'] = nn.Linear(layer_sizes[-1], n_features)
            self.output_activations['n'] = nn.ReLU()
            self.output_layers['p'] = nn.Linear(layer_sizes[-1], n_features)
            self.output_activations['p'] = nn.Sigmoid()
        
        elif(generative_model == 'bernoulli'):
            self.output_layers['p'] = nn.Linear(layer_sizes[-1], n_features)
            self.output_activations['p'] = nn.Sigmoid()
        
        # set initial mask for autoregressive layer weights
        self.update_autoregressive_mask()

    def update_autoregressive_mask(self):
        """
        Updates autoregressive mask applied to final layer weight matrix.
        """
        if(self.is_autoregressive):
            # random binary mask by permutation
            mask = torch.triu(torch.ones((self.n_features,self.n_features)), diagonal=1)
            permute_idx = torch.randperm(self.n_features)
            mask = mask[permute_idx,:]

            # apply mask to each output linear layer weight matrix
            for output_layer in self.output_layers.values():
                output_layer.register_buffer('autoregressive_mask', mask)
                # output_layer.register_buffer('weight_old', output_layer.weight)
                output_layer.weight = nn.Parameter(output_layer.autoregressive_mask * output_layer.weight)
        else:
            pass

    def forward(self, x, condition_labels=None):
        """
        """
        # append condition labels if relevant (may be continuous)
        if(self.n_conditions>0):
            x = torch.concat((x, condition_labels), dim=1)

        # pass sample from latent space through hidden layers
        for model_layer in self.model_layers:
            x = model_layer(x)

        # produce parameters of generative model
        outs = []
        for (param, output_layer) in self.output_layers.items():
            x_out = output_layer(x)
            if(param in self.output_activations.keys()):
                x_out = self.output_activations[param](x_out)
            outs.append(x_out)
        
        return outs

class VAE(nn.Module):
    def __init__(self, n_features, z_dim, layer_sizes, activation=None, generative_model='gaussian', n_batches=1, kl_weight=1e-3, is_autoregressive=False, n_conditions=0, seed=0):
        """
        n_features (int): number of input features per observation
        z_dim (int): number of latent variables
        layer_size (list): number of hidden units for each encoder layer
        activation:
        generative_model (string): data generative model, parametrized by decoder neural network ('gaussian', 'bernoulli', 'nb', 'zinb')
        n_batches (int): number of batches present in data (default 1)
        is_autoregressive (bool): 
        is_conditional (bool):
        seed (int): seed for random number generator
        """
        super(VAE, self).__init__()
        self.input_dim = n_features
        self.z_dim = z_dim
        self.generative_model = generative_model
        self.kl_weight = kl_weight
        self.is_autoregressive = is_autoregressive
        self.n_conditions = n_conditions

        # define encoder network
        self.encoder = Encoder(n_features, z_dim, layer_sizes, n_conditions)

        # define decoder network
        self.decoder = Decoder(z_dim, n_features, layer_sizes, generative_model, is_autoregressive, n_conditions)

        # latent batch offset vector
        if(n_batches>1):
            self.batch_offset = nn.parameter.Parameter(torch.rand(z_dim, n_batches))
        else:
            self.batch_offset = torch.zeros(z_dim, n_batches) # no batch correction

    def generate_data(self, params):
        if(self.generative_model == 'bernoulli'):
            return torch.bernoulli(input=params[0])
        
        elif(self.generative_model == 'gaussian'):
            return torch.normal(mean=params[0], std=torch.exp(0.5*params[1]))
        
        elif(self.generative_model == 'nb'):
            return torch.negative_binomial(n=params[0], p=params[1])

    def calculate_nll(self, x, x_out, params):
        if(self.generative_model == 'bernoulli'):
            return F.binary_cross_entropy(params[0], x, reduction='mean')
            # return -torch.sum(x * torch.log(x_out) + (1-x) * torch.log(1-x_out))
        
        elif(self.generative_model == 'gaussian'):
            return -torch.exp(-0.5*(x - params[0])**2 / params[1]) / (2*params[1]*torch.pi)
        
        elif(self.generative_model == 'nb'):
            pass

    def elbo_loss(self, z_means, z_logsigmas, generative_params, x, x_out):
        """
        """
        # KL divergence between true and approximate prior
        # equivalent to a regularization term
        # see eq. 10 from Kingma and Welling, 2014
        KL_divergence = -(0.5 * torch.mean(torch.sum(1 + z_logsigmas - z_means**2 - torch.exp(z_logsigmas), dim=-1)))

        # negative log-likelihood of generated data
        # equivalent to reconstruction loss
        nll = self.calculate_nll(x, x_out, generative_params)

        # negative KL because we are minimizing loss
        return (nll, KL_divergence)

    def forward(self, x, condition_labels=None):
        # append condition labels if relevant (may be continuous)
        if(self.n_conditions>0):
            x = torch.concat((x, condition_labels), dim=1)

        # multivariate Gaussian prior
        # generate means and variances for latent variables z
        (means, logsigmas) = self.encoder(x)
        stds = torch.exp(0.5*logsigmas)
        self.z = means + stds * torch.normal(mean=0, std=1, size=stds.shape) #+ self.batch_offset

        # generate observation based on sample from latent distribution
        generative_model_params = self.decoder(self.z, condition_labels)

        # generate data
        x_out = self.generate_data(generative_model_params)

        # calculate loss
        (nll, KL_divergence) = self.elbo_loss(means, logsigmas, generative_model_params, x, x_out)
        elbo = nll + self.kl_weight * KL_divergence

        return x_out, generative_model_params, elbo