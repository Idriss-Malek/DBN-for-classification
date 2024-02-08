import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, layer_sizes, latent_dim):
        """
        Variational Autoencoder (VAE) constructor.
        
        Args:
        - layer_sizes (list): List of integers representing the sizes of layers.
        - latent_dim (int): Dimension of the latent space.
        """
        super(VAE, self).__init__()
        self.layer_sizes = layer_sizes
        self.latent_dim = latent_dim
        self.device = torch.device('cpu')

        encoder_layers = []
        for i in range(len(layer_sizes) - 1):
            encoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        self.mu_layer = nn.Linear(layer_sizes[-1], latent_dim)
        self.logvar_layer = nn.Linear(layer_sizes[-1], latent_dim)

        reversed_layer_sizes = layer_sizes[::-1]
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, reversed_layer_sizes[0]))
        for i in range(len(reversed_layer_sizes) - 2):
            decoder_layers.append(nn.Linear(reversed_layer_sizes[i], reversed_layer_sizes[i+1]))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(reversed_layer_sizes[-2], reversed_layer_sizes[-1]))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """
        Encode input data into the latent space.
        
        Args:
        - x (Tensor): Input tensor.
        
        Returns:
        - Tensor: Mean of the latent space.
        - Tensor: Logvariance of the latent space.
        """
        out = self.encoder(x)
        mu = self.mu_layer(out)
        logvar = self.logvar_layer(out)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterize the latent space for sampling.
        
        Args:
        - mu (Tensor): Mean of the latent space.
        - logvar (Tensor): Logvariance of the latent space.
        
        Returns:
        - Tensor: Reparameterized latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decode latent space into the original space.
        
        Args:
        - z (Tensor): Latent space tensor.
        
        Returns:
        - Tensor: Decoded output tensor.
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Forward pass of the VAE.
        
        Args:
        - x (Tensor): Input tensor.
        
        Returns:
        - Tensor: Decoded output tensor.
        - Tensor: Mean of the latent space.
        - Tensor: Logvariance of the latent space.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def generate_images(self, nb_images, size_img):
        """
        Generate images from random samples in the latent space.
        
        Args:
        - nb_images (int): Number of images to generate.
        - size_img (tuple): Size of the images.
        
        Returns:
        - list: List of generated images.
        """
        images = []
        with torch.no_grad():
            for _ in range(nb_images):
                v=torch.randn(self.latent_dim).to(self.device)
                image = self.decode(v).cpu().numpy().reshape(size_img)
                images.append(image)
        return images
