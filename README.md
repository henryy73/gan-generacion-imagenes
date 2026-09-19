# Facial Image Synthesis: DCGAN vs. WGAN-GP

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)

This repository compares DCGAN and WGAN-GP architectures for facial image synthesis on the CelebA dataset. It evaluates training stability, convergence, and generation quality, using the Fréchet Inception Distance (FID) metric across epochs to provide an objective measure of generative performance.

## 📂 Repository Structure

The project follows a modular structure, separating the heavy mathematical training logic, the theoretical foundations, and the visual analysis:

```text
├── docs/                     
│   └── Mathematical_Foundations.pdf  # Theoretical analysis and mathematical proofs
├── dcgan.py                          # Implementation of standard DCGAN with Spectral Normalization
├── wgan_gp.py                        # Implementation of WGAN with Gradient Penalty
├── Model_evaluation.ipynb            # Interactive notebook comparing precomputed results
├── results_dcgan/                    # Precomputed metrics and plots for DCGAN
│   └── training_metrics.npz
├── results_wgan/                     # Precomputed metrics and plots for WGAN-GP
│   └── training_metrics.npz
└── README.md                         # Project documentation
