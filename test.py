import torch
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def test(dnn, X, y):
    """
    Test a deep neural network model on the given data.

    Args:
    - dnn (DNN): The DNN model to test.
    - X (Tensor): Input data tensor.
    - y (Tensor): Target labels tensor.

    Returns:
    - float: Accuracy of the model on the given data.
    """
    X = X.to(dnn.device)
    outputs = dnn(X)
    _, y_pred = torch.max(outputs, dim=1)
    accuracy = accuracy_score(y, y_pred.cpu())
    return accuracy

def save_images(images, nrows, ncols, file_name):
    """
    Save images to a file.

    Args:
    - images (list): List of images to save.
    - nrows (int): Number of rows in the plot grid.
    - ncols (int): Number of columns in the plot grid.
    - file_name (str): File name to save the plot.

    Returns:
    - None
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4))
    axes = axes.flatten()
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(file_name)

def display_images(images, nrows, ncols):
    """
    Display images in a grid.

    Args:
    - images (list): List of images to display.
    - nrows (int): Number of rows in the plot grid.
    - ncols (int): Number of columns in the plot grid.

    Returns:
    - None
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4))
    axes = axes.flatten()
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def read_mnist(data, shuffle=True):
    """
    Load MNIST data.

    Args:
    - data (dict): Dictionary containing MNIST data.
    - shuffle (bool): Whether to shuffle the data.

    Returns:
    - tuple: Tuple containing train and test data and labels.
    """
    X_train = np.round(np.concatenate([data[f'train{i}'] for i in range(10)]) / 255)
    y_train = np.concatenate([i * np.ones(data[f'train{i}'].shape[0]) for i in range(10)])
    X_test = np.round(np.concatenate([data[f'test{i}'] for i in range(10)]) / 255)
    y_test = np.concatenate([i * np.ones(data[f'test{i}'].shape[0]) for i in range(10)])
    indices = np.arange(y_train.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    return torch.tensor(X_train[indices]).float(), torch.tensor(y_train[indices]).long(), torch.tensor(
        X_test).float(), torch.tensor(y_test).long()


def read_alpha_digit(data, idx, shuffle=False):
    """
    Load alpha digit data.

    Args:
    - data (dict): Dictionary containing alpha digit data.
    - idx (list): List of indices to select from the data.
    - shuffle (bool): Whether to shuffle the data.

    Returns:
    - Tensor: Data tensor.
    """
    X = data['dat'][idx[0]]
    n = X.shape[0] * len(idx)
    p = X[0].shape[0] * X[0].shape[1]
    for i in range(1, len(idx)):
        X_bis = data['dat'][idx[i]]
        X = np.concatenate((X, X_bis), axis=0)
    X = np.concatenate(X).reshape((n, p))
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    return torch.tensor(X[indices]).float()

