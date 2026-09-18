# gan-generacion-imagenes
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

This repository compares DCGAN and WGAN-GP architectures for facial image synthesis on the CelebA dataset. It evaluates training stability, convergence, and generation quality, using the Fréchet Inception Distance (FID) metric across epochs to provide an objective measure of generative performance.

## 📂 Repository Structure

```text
├── dcgan.py             # Implementation of standard DCGAN with Spectral Normalization
├── wgan-gp.py           # Implementation of WGAN with Gradient Penalty (Critic and GP logic)
├── README.md            # Project documentation
└── assets/              # Folder containing saved plots and generated images
