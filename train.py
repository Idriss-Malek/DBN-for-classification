import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def train_RBM(rbm, X, learning_rate = 0.01, batch_size = 32, n_epochs = 100, verbose=True, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Train a single Restricted Boltzmann Machine (RBM).

    Args:
    - rbm (RBM): The RBM model to train.
    - X (Tensor): Input data tensor.
    - learning_rate (float): Learning rate for the training.
    - batch_size (int): Batch size for mini-batch training.
    - n_epochs (int): Number of epochs for training.
    - verbose (bool): Whether to print training progress.
    - device (torch.device): Device to perform training.

    Returns:
    - None
    """
    rbm.to(device)
    rbm.device = device
    for i in range(1, n_epochs + 1):
        X = X.to(device)
        n=X.shape[0]
        for i in range(0, n ,batch_size):
            X_batch=X[i:min(i+batch_size,n),:]
            batch_size = X_batch.shape[0] 

            V = X_batch
            proba_H = rbm.input_to_latent(V)
            H = (torch.rand(batch_size,rbm.out_features).to(device) < proba_H).float()
            proba_V2 = rbm.latent_to_input(H)
            V2 = (torch.rand(batch_size,rbm.in_features).to(device) < proba_V2).float()
            proba_H2 = rbm.input_to_latent(V2)

            da = torch.sum(V-V2,dim=0)
            db = torch.sum(proba_H-proba_H2,dim=0)
            dW = proba_H.T@V - proba_H2.T@V2
            with torch.no_grad():
                detached_a = rbm.a.detach()
                detached_bias = rbm.bias.detach()
                detached_weight = rbm.weight.detach()
                detached_a = detached_a + learning_rate * da
                detached_bias = detached_bias + learning_rate * db
                detached_weight = detached_weight + learning_rate * dW
                rbm.a.data.copy_(detached_a)
                rbm.bias.data.copy_(detached_bias)
                rbm.weight.data.copy_(detached_weight)
        H = rbm.input_to_latent(X)
        X_res = rbm.latent_to_input(H)
        loss = F.mse_loss(X_res, X)
        if i%10 == 0 and verbose:
            print(f'Epoch : {i}. Loss : {loss}.')

def train_DBN(dbn, X, learning_rate = 0.01, batch_size = 32, n_epochs = 100, verbose=True, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Pretrain a Deep Belief Network (DBN) layer by layer using RBMs.

    Args:
    - dbn (DBN): The DBN model to pretrain.
    - X (Tensor): Input data tensor.
    - learning_rate (float): Learning rate for the training.
    - batch_size (int): Batch size for mini-batch training.
    - n_epochs (int): Number of epochs for training.
    - verbose (bool): Whether to print training progress.
    - device (torch.device): Device to perform training.

    Returns:
    - None
    """
    dbn.device = device
    X = X.to(device)
    for i,rbm in enumerate(dbn.rbms):
        train_RBM(rbm, X, learning_rate, batch_size, n_epochs, False, device)
        H = rbm(X)
        if verbose:
            X_res = rbm.latent_to_input(H)
            loss = F.mse_loss(X_res, X)
            print(f'RBM {i+1} pretrained. Loss is {loss}.')
        X = H

def pretrain(dnn,X, learning_rate = 0.01, batch_size = 32, n_epochs = 100, verbose=True, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Pretrain a deep neural network (DNN) using DBN pretraining.

    Args:
    - dnn (DNN): The DNN model to pretrain.
    - X (Tensor): Input data tensor.
    - learning_rate (float): Learning rate for the training.
    - batch_size (int): Batch size for mini-batch training.
    - n_epochs (int): Number of epochs for training.
    - verbose (bool): Whether to print training progress.
    - device (torch.device): Device to perform training.

    Returns:
    - None
    """
    train_DBN(dnn.dbn, X, learning_rate, batch_size, n_epochs, verbose, device)

def backpropagation(dnn, X, y, learning_rate=0.01, batch_size=32, n_epochs=200, verbose=True, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Train a DNN using backpropagation with mini-batch gradient descent.

    Args:
    - dnn (DNN): The DNN model to train.
    - X (Tensor): Input data tensor.
    - y (Tensor): Target labels tensor.
    - learning_rate (float): Learning rate for the training.
    - batch_size (int): Batch size for mini-batch training.
    - n_epochs (int): Number of epochs for training.
    - verbose (bool): Whether to print training progress.
    - device (torch.device): Device to perform training.

    Returns:
    - None
    """
    for i,rbm in enumerate(dnn.dbn.rbms):
        rbm.to(device)
    dnn.to(device)
    dnn.device = device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dnn.parameters(), lr=learning_rate)
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for epoch in range(n_epochs):
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            dnn.train()
            optimizer.zero_grad()
            outputs = dnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if verbose and epoch%10 == 0: 
            dnn.eval()
            outputs= dnn(X)
            CELoss = criterion(outputs,y)
            print(f'Epoch : {epoch} | Loss : {CELoss}')

def vae_loss(recon_x, x, mu, logvar):
    """
    Calculate the loss for training a Variational Autoencoder (VAE).

    Args:
    - recon_x (Tensor): Reconstructed input tensor.
    - x (Tensor): Original input tensor.
    - mu (Tensor): Mean of the latent space.
    - logvar (Tensor): Logvariance of the latent space.
        Returns:
    - Tensor: VAE loss.

    """
    recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

def train_VAE(vae, X, learning_rate=0.01, batch_size=32, n_epochs=200, verbose=True, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Train a Variational Autoencoder (VAE).

    Args:
    - vae (VAE): The VAE model to train.
    - X (Tensor): Input data tensor.
    - learning_rate (float): Learning rate for the training.
    - batch_size (int): Batch size for mini-batch training.
    - n_epochs (int): Number of epochs for training.
    - verbose (bool): Whether to print training progress.
    - device (torch.device): Device to perform training.

    Returns:
    - None
    """   
    vae.to(device)
    vae.device = device
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for epoch in range(n_epochs):
        for batch in dataloader:
            x = batch[0].to(device)
            vae.train()
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(x)
            loss = vae_loss(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
        if verbose and epoch%10 == 0: 
            vae.eval()
            recon_X, mu, logvar = vae(X)
            vaeloss = vae_loss(recon_X, X, mu, logvar)
            print(f'Epoch : {epoch} | Loss : {vaeloss}')
