# WCGAN-for-Galaxy-Spectrum
This repository contains the project code developed during the Asian Astro AI Workshop in Osaka, Japan.

# Structure of Model
## Generator 
The Generator starts with a Multi-Layer Perceptron (MLP) that includes two hidden layers to extract initial features. Following the MLP, we use a ResNet-18 architecture, with an opti

## Discriminator
The Discriminator consists of three convolutional layers, each followed by a max pooling layer. After the convolutional layers, a fully connected layer is used to differentiate between real and generated spectra.

# Training Script for WCGAN
This script, Training.py, implements the training process for a Wasserstein Conditional GAN (WCGAN) designed to generate galaxy spectra. The WCGAN architecture combines the stability of the Wasserstein GAN with the conditioning on input parameters to generate realistic and diverse spectra based on specific physical conditions.

## Key Features:
***Wasserstein Loss*** The training leverages the Wasserstein loss function, which provides a more stable and effective training process by minimizing the Wasserstein distance between the real and generated distributions
***Conditional GAN***
The model is conditioned on input parameters, allowing it to generate spectra that adhere to the specified conditions, enhancing the model's controllability and precision.
## Usage
The training process is controlled by a set of hyperparameters, which can be adjusted to optimize performance. The script outputs the trained models, loss metrics, and sample spectra at various epochs, providing insights into the model's progress and effectiveness.

# Visualizable Training
You can see the information in Tensorboard. Addtionally, it support user add more information during the training by adding the tensorboard script.
