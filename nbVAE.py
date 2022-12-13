import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, NegativeBinomial
from torch.distributions import kl_divergence as kl

# Define the encoder module (INFERENCE)
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_dims, latent_dims):
        super(Encoder, self).__init__()

        # First hidden layer
        self.linear1 = nn.Linear(input_size, hidden_dims)
        # Layer for mu
        self.linear_mu = nn.Linear(hidden_dims, latent_dims)
        # Layer for (log) sigma
        self.linear_sigma = nn.Linear(hidden_dims, latent_dims)

        # Store standard Normal distribution
        self.N = Normal(0,1)

    def forward(self, x):
        # Calculate total gene counts for each cell
        library = torch.sum(x, dim=1, keepdim=True)

        # Send the data through the first hidden layer
        h_ = F.relu(self.linear1(torch.log(1+x)))

        # Calculate latent mu from transformed data
        mu =  self.linear_mu(h_)
        # Calculate latent sigma from transformed data
        sigma = torch.exp(self.linear_sigma(h_))

        # Re-parameterize the latent variables as a standard Normal
        z = mu + sigma*self.N.sample(mu.shape)

        return z, mu, sigma, library

# Define the decoder module (GENERATIVE)
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_dims, latent_dims):
        super(Decoder, self).__init__()

        # First hidden layer
        self.linear1 = nn.Linear(latent_dims, hidden_dims)
        # Layer for negative-binomial scale parameter
        self.linear_scale = nn.Linear(hidden_dims, output_size)

        # Treat theta as a learned parameter
        self.log_theta = torch.nn.Parameter(torch.randn(output_size))

    def forward(self, z, library):
        # Send the latent data representation through the first hidden layer
        h_ = F.relu(self.linear1(z))

        # Calculate scale parameter for the data distribution
        scale = torch.sigmoid(self.linear_scale(h_))
        # The rate parameter is just the scale multiplied by the total library size for each cell
        # (each gene per cell has its own rate parameter, 
        # which will help determine the probability of a success for each trial)
        rate = scale * library
        # Theta is the number of "failures" after which to stop our Bernoulli trials
        theta = torch.exp(self.log_theta)

        return rate, theta

# Define the full VAE
class VAE(nn.Module):
    def __init__(self, input_size, hidden_dims=128, latent_dims=10):
        super(VAE, self).__init__()

        # Define Encoder and Decoder modules
        self.encoder = Encoder(input_size, hidden_dims, latent_dims)
        self.decoder = Decoder(input_size, hidden_dims, latent_dims)

    def forward(self, x):
        # Perform inference on x using Encoder to generate latent z & params
        z, mu, sigma, library = self.encoder(x)
        # Generate negative binomial parameters using latent z
        rate, theta = self.decoder(z, library)
        
        # Calculate the KL-divergence between a standard normal and the variational distribution of z
        prior_dist = Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        var_post_dist = Normal(mu, torch.sqrt(sigma)+1e-4)
        kl_div = kl(var_post_dist, prior_dist).sum(dim=1)

        # Calculate the negative binomial log-likelihood using our generated parameters and the data x
        nb_logits = (rate + 1e-4).log() - (theta + 1e-4).log()
        log_lik = NegativeBinomial(total_count=theta, logits=nb_logits).log_prob(x).sum(dim=-1)

        # Calculate the ELBO & loss from the log-likelihood and the KL-divergence
        elbo = log_lik - kl_div
        loss = torch.mean(-elbo)

        return loss
    
    def reconstruct_data(self, x):
        # Perform inference on x using Encoder to generate latent z & params
        z, _, _, library = self.encoder(x)
        # Generate negative binomial parameters using latent z
        rate, theta = self.decoder(z, library)

        # Sample from a negative binomial with our decoded parameters to reconstruct the input x
        nb_logits = (rate + 1e-4).log() - (theta + 1e-4).log()
        reconstruction = NegativeBinomial(total_count=theta, logits=nb_logits).sample()

        return reconstruction
    
    def get_latents(self, x):
        # Perform inference on x using Encoder to generate latent z & params
        z, _, _, _ = self.encoder(x)

        return z