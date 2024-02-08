# Deep Belief Networks for Classification

This repository demonstrates the utilization of Deep Belief Networks (DBNs) to significantly enhance the accuracy of Multilayer Perceptrons (MLPs) on classification tasks.

## Repository Contents:

- **rbm.py**: Implementation of the Restricted Boltzmann Machine (RBM) class.
- **dbn.py**: Implementation of the Deep Belief Network (DBN) class.
- **dnn.py**: Implementation of the Deep Neural Network (DNN) class, comprising a DBN followed by a classification layer.
- **vae.py**: Implementation of the Variational Autoencoder (VAE) class.
- **train.py**: Contains training functions:
  - `train_RBM`: Unsupervised training for RBM.
  - `train_DBN`: Unsupervised training for DBN.
  - `pretrain`: Unsupervised training for the DBN within a DNN.
  - `backpropagation`: Standard training loop for DNN.
  - `train_VAE`: Standard training loop for VAE.
- **test.py**: Contains testing functions:
  - `test`: Computes the accuracy of a model on testing samples.
  - `save_images`: Saves images in a folder.
  - `display_images`: Displays images.
  - `read_mnist`: Loads the MNIST dataset.
  - `read_alpha_digit`: Loads the BinaryAlphaDigit dataset.


## Data Sources:
To use `read_mnist` and `read_alpha_digit` you need to download the datasets from [THIS CLICKABLE LINK](https://cs.nyu.edu/~roweis/data.html).
- **MNIST Dataset**: To load the MNIST dataset, utilize the `read_mnist` function provided in `test.py`.
- **BinaryAlphaDigit Dataset**: To load the BinaryAlphaDigit dataset, use the `read_alpha_digit` function provided in `test.py`.


**Note**: Ensure you have necessary dependencies installed before running the scripts. You will need PyTorch to use the models.
