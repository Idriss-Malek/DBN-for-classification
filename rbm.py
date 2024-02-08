import torch
import torch.nn as nn
import torch.nn.functional as F

class RBM(nn.Module):
    def __init__(self, in_features, out_features):
        """
        Restricted Boltzmann Machine (RBM) constructor.
        
        Args:
        - in_features (int): Number of input features.
        - out_features (int): Number of output features.
        """
        super(RBM, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.a = nn.Parameter(torch.zeros(in_features))
        self.device = torch.device('cpu')
        
    def forward(self, input):
        """
        Forward pass of the RBM.
        
        Args:
        - input (Tensor): Input tensor.
        
        Returns:
        - Tensor: Output tensor after applying sigmoid activation.
        """
        return torch.sigmoid(F.linear(input, self.weight, self.bias))
         
    def input_to_latent(self,input):
        """
        Convert input to latent representation.
        
        Args:
        - input (Tensor): Input tensor.
        
        Returns:
        - Tensor: Latent representation tensor.
        """
        return self(input)
    
    def latent_to_input(self,input):
        """
        Convert latent representation to input.
        
        Args:
        - input (Tensor): Latent representation tensor.
        
        Returns:
        - Tensor: Reconstructed input tensor.
        """
        return torch.sigmoid(F.linear(input, self.weight.transpose(0,1), self.a))
    
    def extra_repr(self):
        """
        Extra representation of RBM module.
        """
        return 'RBM : in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
    
    def generate_images(self, nb_images, nb_iter, size_img):
        """
        Generate images using Gibbs sampling.
        
        Args:
        - nb_images (int): Number of images to generate.
        - nb_iter (int): Number of Gibbs sampling iterations.
        - size_img (tuple): Size of the images.
        
        Returns:
        - list: List of generated images.
        """
        images = []
        device = self.device
        for _ in range(nb_images):
            v=(torch.rand(self.in_features)<0.5).to(self.device).float()
            for __ in range(nb_iter):
                h = (torch.rand(self.out_features).to(self.device)<self.input_to_latent(v)).float()
                v = (torch.rand(self.in_features).to(self.device)<self.latent_to_input(h)).float()
            v=v.cpu().numpy().reshape(size_img)
            images.append(v)
        return images
