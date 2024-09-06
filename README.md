# WCGAN-for-Galaxy-Spectrum
This repository contains the project code developed during the Asian Astro AI Workshop in Osaka, Japan.

# Structure of Model
The script, **Model.py**, describe the more useful model structure of this project (**We get the result with gan.py**). The cbam residual block is described in **ResNet.py**.[Residual Connection](https://arxiv.org/abs/1512.03385) [GAN](https://arxiv.org/abs/1406.2661)
## Generator 
The Generator starts with a Multi-Layer Perceptron (MLP) that includes two hidden layers to extract initial features. Following the MLP, we use a ResNet-18 architecture, with an optional CBAM attention mechanism. [CBAM](https://arxiv.org/pdf/1807.06521)

## Discriminator
The Discriminator consist with the MLP and cbamResNet, we don't use any batch normliaze on discriminator

# Training Script for WCGAN
We use Wasserstein Loss to fit model. For reference [WGAN](https://arxiv.org/abs/1704.00028)

## Key Features:
***Wasserstein Loss*** The training leverages the Wasserstein loss function, which provides a more stable and effective training process by minimizing the Wasserstein distance between the real and generated distributions.

***Conditional GAN***
The model is conditioned on input parameters, allowing it to generate spectra that adhere to specified physical conditions, (more close to semi-unsupervised learning)thus enhancing the model's controllability and precision. Unlike traditional conditional GANs【Paper (https://arxiv.org/pdf/1411.1784)】, our model is specifically designed to generate spectra based on parameters that may be useful for observational research. The generator takes in four physical parameters—redshift, age, metallicity, and solar mass—as input. These parameters act similarly to noise, guiding the generation of spectra.

The discriminator's role is to distinguish whether the generated spectra are real or fake. Notably, the discriminator does not require the input parameters during training, as our goal is to predict spectra that could plausibly exist in the universe, even if they haven't been observed yet.
## Usage
Now the trained model has store in `model` file. Source code are in `gan.py`. `argparseinfo` store some useful hyperparameter if you wnat to model more advanced and efficient.


