import torch
import torch.nn as nn

from rbm import RBM

class DBN(nn.Module):
    def __init__(self, layer_sizes):
        """
        Deep Belief Network (DBN) constructor.
        
        Args:
        - layer_sizes (list): List of integers representing the sizes of layers.
        """
        super(DBN, self).__init__()
        self.layer_sizes = layer_sizes
        self.rbms = [RBM(layer_sizes[i],layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
        self.device = torch.device('cpu')
        
    def forward(self, input):
        """
        Forward pass of the DBN.
        
        Args:
        - input (Tensor): Input tensor.
        
        Returns:
        - Tensor: Output tensor after passing through the DBN.
        """
        out = input
        for rbm in self.rbms:
            out = rbm(out)
        return out
    
    def extra_repr(self):
        """
        Extra representation of DBN module.
        """
        return 'DBN : layer_sizes={}'.format(
            self.layer_sizes
        )
    
    def generate_images(self, nb_images, nb_iter, size_img):
        """
        Generate images using Gibbs sampling in the DBN.
        
        Args:
        - nb_images (int): Number of images to generate.
        - nb_iter (int): Number of Gibbs sampling iterations.
        - size_img (tuple): Size of the images.
        
        Returns:
        - list: List of generated images.
        """
        images = []
        for _ in range(nb_images):
            v=(torch.rand(self.layer_sizes[0])<0.5).to(self.device).float()
            for __ in range(nb_iter):
                for i in range(len(self.rbms)):
                    v = (torch.rand(self.layer_sizes[i+1]).to(self.device)<self.rbms[i].input_to_latent(v)).float()
                for i in range(len(self.rbms)-1,-1,-1):
                    v = (torch.rand(self.layer_sizes[i]).to(self.device)<self.rbms[i].latent_to_input(v)).float()
            v=v.cpu().numpy().reshape(size_img)
            images.append(v)
        return images
