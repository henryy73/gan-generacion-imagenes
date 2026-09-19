# ==========================================
# Standard PyTorch and utility imports
# ==========================================
import os
import time
import random
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as models
from torchvision.models import Inception_V3_Weights

# Record the start time of the entire script
start_time = time.time()

# Set environment variable for deterministic algorithms (useful for reproducibility)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# ==========================================
# Experiment Configuration
# ==========================================
CONFIG = {
    "dataroot": "C:/Users/ematm/Downloads/img_align_celeba", # Relative path to the extracted dataset folder
    "workers": 4,             # Number of worker threads for loading data
    "batch_size": 128,        # Number of images in a single training batch
    "image_size": 64,         # Spatial size of training images (resized to 64x64)
    "nc": 3,                  # Number of color channels (3 for RGB)
    "nz": 100,                # Size of the latent z vector (generator input noise)
    "ngf": 64,                # Size of feature maps in the generator
    "ndf": 64,                # Size of feature maps in the discriminator/critic
    "num_epochs": 30,         # Number of training epochs (10 is usually enough for WGAN-GP to show results)
    "lr": 0.0001,             # Learning rate (standard for WGAN-GP)
    "beta1": 0.0,             # Beta1 hyperparameter for Adam optimizer (WGAN-GP strictly requires 0.0)
    "beta2": 0.9,             # Beta2 hyperparameter for Adam optimizer
    "lambda_gp": 10,          # Gradient penalty weight (standard value from the WGAN-GP paper)
    "n_critic": 5,            # Number of critic updates per generator update
    "ngpu": 1,                # Number of GPUs to use (0 for CPU)
    "seed": 999,              # Random seed for reproducibility
    "save_dir": "./checkpoints", # Directory to save model weights and generated plots
    "fid_samples": 1000       # Number of samples to use for FID calculation (1000 for speed, 10000+ for accuracy)
}

# ==========================================
# Reproducibility and directory setup
# ==========================================
random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
torch.use_deterministic_algorithms(False) # Set to True if strict reproducibility is needed
os.makedirs(CONFIG["save_dir"], exist_ok=True) # Create output directory if it doesn't exist
torch.backends.cudnn.benchmark = True # Optimizes performance if input sizes don't change

# ==========================================
# FID Calculation & Inception Feature Extractor
# ==========================================
class InceptionFeatureExtractor(nn.Module):
    """Extracts features using a pre-trained Inception V3 network for FID calculation."""
    def __init__(self, device):
        super(InceptionFeatureExtractor, self).__init__()
        self.device = device
        # Load pre-trained Inception v3
        self.model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.model.fc = nn.Identity() # Remove the final classification layer to get raw features
        self.model.eval()             # Set model to evaluation mode
        self.model.to(self.device)

    def forward(self, x):
        # Scale inputs from [-1, 1] (GAN output) to [0, 1] (Inception expected input)
        x = (x + 1.0) / 2.0
        # Resize images to 299x299 as required by Inception v3
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        with torch.no_grad():
            features = self.model(x)
        return features

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculates the Frechet Inception Distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    # Calculate the square root of the dot product of the covariance matrices
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Handle numerical instabilities
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    tr_covmean = np.trace(covmean)
    # FID Formula: ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def calculate_fid(netG, dataloader, feature_extractor, config, device, num_samples):
    """Generates fake images, compares them to real images, and calculates the FID score."""
    print(f"\nCalculating FID over {num_samples} samples...")
    netG.eval() # Set generator to evaluation mode
    real_features = []
    fake_features = []
    collected = 0
    
    for data in dataloader:
        real_imgs = data[0].to(device)
        b_size = real_imgs.size(0)
        
        # Stop collecting if we reach the target number of samples
        if collected + b_size > num_samples:
            real_imgs = real_imgs[:num_samples - collected]
            b_size = real_imgs.size(0)
            
        # Extract features for real images
        with torch.no_grad():
            real_feat = feature_extractor(real_imgs)
        real_features.append(real_feat.cpu().numpy())
        
        # Generate fake images and extract their features
        noise = torch.randn(b_size, config["nz"], 1, 1, device=device)
        with torch.no_grad():
            fake_imgs = netG(noise)
            fake_feat = feature_extractor(fake_imgs)
        fake_features.append(fake_feat.cpu().numpy())
        
        collected += b_size
        if collected >= num_samples:
            break
            
    netG.train() # Return generator to training mode
    
    # Concatenate all features into single arrays
    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)
    
    # Calculate mean and covariance for real and fake feature distributions
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    
    # Compute and return the final FID score
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid

# ==========================================
# 1. Data Loading
# ==========================================
# Load the dataset and apply transformations (Resize, Crop, Tensor conversion, and Normalization to [-1, 1])
dataset = dset.ImageFolder(root=CONFIG["dataroot"],
                           transform=transforms.Compose([
                               transforms.Resize(CONFIG["image_size"]),
                               transforms.CenterCrop(CONFIG["image_size"]),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

# Create the dataloader to serve batches of data during training
dataloader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG["batch_size"],
                                         shuffle=True, num_workers=CONFIG["workers"], pin_memory=True)

# Select GPU if available, otherwise fallback to CPU
device = torch.device("cuda:0" if (torch.cuda.is_available() and CONFIG["ngpu"] > 0) else "cpu")
print(f"Running on device: {device}")


# ==========================================
# 2. Model Definition & Weight Initialization
# ==========================================
def weights_init(m):
    """Custom weights initialization called on netG and netD."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    """The Generator takes a noise vector and upsamples it into an image."""
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        nz, ngf, nc = CONFIG["nz"], CONFIG["ngf"], CONFIG["nc"]
        
        self.main = nn.Sequential(
            # Input is Z (noise), going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State size: (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Final state size: (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Instantiate the Generator and apply weight initialization
netG = Generator(CONFIG["ngpu"]).to(device)
netG.apply(weights_init)

class Critic(nn.Module):
    """In WGAN-GP, the Discriminator is called a Critic because it outputs a continuous score, not a probability."""
    def __init__(self, ngpu):
        super(Critic, self).__init__()
        self.ngpu = ngpu
        nc, ndf = CONFIG["nc"], CONFIG["ndf"]
        
        self.main = nn.Sequential(
            # NOTE: No BatchNorm2d is used in the Critic because it breaks Gradient Penalty math.
            # Input size: (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf*8) x 4 x 4
            # Outputs a single raw score (no Sigmoid at the end for WGAN)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        # Flatten the output to a 1D vector per image in the batch
        return self.main(input).view(input.size(0), -1)

# Instantiate the Critic and apply weight initialization
netD = Critic(CONFIG["ngpu"]).to(device)
netD.apply(weights_init)

# ==========================================
# WGAN-GP Gradient Penalty Function
# ==========================================
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN-GP to enforce the Lipschitz constraint."""
    # Generate random weights for interpolating between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    
    # Create random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    
    # Create tensor of ones to match d_interpolates size (required for autograd.grad)
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
    
    # Calculate the gradients of the critic's output with respect to the interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # Flatten the gradients
    gradients = gradients.view(gradients.size(0), -1)
    
    # Gradient penalty formula: ((L2_norm(gradients) - 1) ** 2).mean()
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ==========================================
# 3. Training Loop (WGAN-GP)
# ==========================================
if __name__ == '__main__':
    # Create a fixed noise vector to track visual progress of the Generator over time
    fixed_noise = torch.randn(64, CONFIG["nz"], 1, 1, device=device)

    # Setup Adam optimizers for both G and D. WGAN-GP relies strictly on beta1=0.0
    optimizerD = optim.Adam(netD.parameters(), lr=CONFIG["lr"], betas=(CONFIG["beta1"], CONFIG["beta2"]))
    optimizerG = optim.Adam(netG.parameters(), lr=CONFIG["lr"], betas=(CONFIG["beta1"], CONFIG["beta2"]))
    
    # Initialize the FID feature extractor
    feature_extractor = InceptionFeatureExtractor(device)

    # Lists to keep track of progress and metrics
    img_list = []
    G_losses = []  
    D_losses = []  
    d_epoch_losses = []  
    g_epoch_losses = []  
    fid_scores = []     
    iters = 0

    print("Starting WGAN-GP Training Loop...")
    for epoch in range(CONFIG["num_epochs"]):
        running_d_loss = 0.0
        running_g_loss = 0.0
        
        for i, data in enumerate(dataloader, 0):
            real_imgs = data[0].to(device)
            b_size = real_imgs.size(0)

            # ==========================================
            # 1. Update Critic (Discriminator)
            # ==========================================
            netD.zero_grad()
            
            # Generate fake images
            noise = torch.randn(b_size, CONFIG["nz"], 1, 1, device=device)
            fake_imgs = netG(noise)

            # Pass real and fake images through the Critic
            real_validity = netD(real_imgs)
            fake_validity = netD(fake_imgs.detach()) # Detach prevents gradients from flowing to G

            # Calculate Gradient Penalty
            gradient_penalty = compute_gradient_penalty(netD, real_imgs.data, fake_imgs.data, device)

            # Critic Loss Formula: (Score of Fakes) - (Score of Reals) + (Gradient Penalty Weight * Penalty)
            errD = torch.mean(fake_validity) - torch.mean(real_validity) + (CONFIG["lambda_gp"] * gradient_penalty)
            errD.backward()
            optimizerD.step()
            
            # Record tracking variables for the Critic
            running_d_loss += errD.item()
            D_x_score = torch.mean(real_validity).item()
            D_G_z_score = torch.mean(fake_validity).item()

            # ==========================================
            # 2. Update Generator (only every n_critic steps)
            # ==========================================
            # WGAN-GP requires the critic to be more heavily trained than the generator
            if i % CONFIG["n_critic"] == 0:
                netG.zero_grad()
                
                # Generate a new batch of fake images for Generator update
                noise_G = torch.randn(b_size, CONFIG["nz"], 1, 1, device=device)
                fake_imgs_G = netG(noise_G)
                
                # Pass fakes through the Critic
                fake_validity_G = netD(fake_imgs_G)
                
                # Generator Loss Formula: - (Score of Fakes)
                # The generator wants the critic to output a HIGH score for fakes
                errG = -torch.mean(fake_validity_G)
                errG.backward()
                optimizerG.step()
                
                # Record tracking variables for the Generator
                running_g_loss += errG.item()
                G_losses.append(errG.item())
                D_losses.append(errD.item())

            # Print training metrics periodically
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_C: %.4f\tLoss_G: %.4f\tReal Score: %.4f\tFake Score: %.4f'
                      % (epoch, CONFIG["num_epochs"], i, len(dataloader),
                         errD.item(), errG.item(), D_x_score, D_G_z_score))


            # Save generated images based on the fixed noise to visualize progress
            if (iters % 500 == 0) or ((epoch == CONFIG["num_epochs"]-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake_grid = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake_grid, padding=2, normalize=True))

            iters += 1

        # Store average losses for this epoch
        d_epoch_losses.append(running_d_loss / len(dataloader))
        # Account for G only updating 1/5th as often when computing the average epoch loss
        g_epoch_losses.append((running_g_loss * CONFIG["n_critic"]) / len(dataloader)) 

        # Calculate FID at the end of the epoch to evaluate generation quality objectively
        current_fid = calculate_fid(netG, dataloader, feature_extractor, CONFIG, device, num_samples=CONFIG["fid_samples"])
        fid_scores.append(current_fid)

        # Save model weights (checkpoints) at the end of each epoch to prevent data loss
        torch.save(netG.state_dict(), os.path.join(CONFIG["save_dir"], f"netG_epoch_{epoch}.pth"))
        torch.save(netD.state_dict(), os.path.join(CONFIG["save_dir"], f"netD_epoch_{epoch}.pth"))
        
        print(f"--- Epoch [{epoch+1}/{CONFIG['num_epochs']}] Summary ---")
        print(f"Avg Critic Loss: {d_epoch_losses[-1]:.4f} | Avg G Loss: {g_epoch_losses[-1]:.4f} | FID: {current_fid:.4f}")
        print(f"Checkpoints saved for epoch {epoch} in {CONFIG['save_dir']}\n")

    # ==========================================
    # 4. Post-Training: Save, Graph, and Visualize
    # ==========================================
    print("Training Complete. Saving metrics and generating plots...")
    
    # Save the numeric tracking arrays for later evaluation
    metrics_path = os.path.join(CONFIG["save_dir"], "training_metrics.npz")
    np.savez(metrics_path, G_losses=G_losses, D_losses=D_losses, fid_scores=fid_scores)
    
    # Plot the Generator and Critic Loss
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Critic Loss During WGAN-GP Training")
    plt.plot(G_losses, label="G Loss", alpha=0.7)
    plt.plot(D_losses, label="Critic Loss", alpha=0.7)
    plt.xlabel("Iterations (G Updates)")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(CONFIG["save_dir"], "loss_plot.png"))
    plt.show()

    # Plot the FID scores across epochs
    plt.figure(figsize=(10, 5))
    plt.title("FID Score Over Epochs")
    epochs_range = range(1, CONFIG["num_epochs"] + 1)
    plt.plot(epochs_range, fid_scores, marker='o', color='red', label="FID")
    plt.xlabel("Epochs")
    plt.ylabel("FID Score")
    plt.xticks(epochs_range)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(CONFIG["save_dir"], "fid_plot.png"))
    plt.show()

    # Display a grid of the final batch of generated images
    print("Displaying final generated images...")
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated Images at Final Epoch")
    final_images = np.transpose(img_list[-1], (1, 2, 0)) # Change dimensions for matplotlib (C, H, W) -> (H, W, C)
    plt.imshow(final_images)
    plt.savefig(os.path.join(CONFIG["save_dir"], "final_generated_images.png"))
    plt.show()

    # ==========================================
    # Display Total Execution Time
    # ==========================================
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    # ==========================================
    # Display Total Execution Time
    # ==========================================
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    # Convert to integers outside the print statement
    h = int(hours)
    m = int(minutes)
    
    print("\n==========================================")
    # Using standard .format() which never confuses linters
    print("Total Execution Time: {:02d}h {:02d}m {:05.2f}s".format(h, m, seconds))
    print("==========================================")