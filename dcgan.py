# ==========================================
# Standard PyTorch and utility imports
# ==========================================
import os
import time  # Added to track total execution time

# Record the start time of the entire script
start_time = time.time()

# Ensure deterministic behavior for cuBLAS operations if needed
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as models
from torchvision.models import Inception_V3_Weights
from torch.nn.utils import spectral_norm # Used to stabilize the Discriminator

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
    "ndf": 64,                # Size of feature maps in the discriminator
    "num_epochs": 30,         # Total number of training epochs
    "beta1": 0.5,             # Beta1 hyperparameter for Adam optimizer (0.5 is standard for DCGAN)
    "ngpu": 1,                # Number of GPUs to use (0 for CPU)
    "seed": 999,              # Random seed for reproducibility
    "save_dir": "./checkpoints" # Directory to save model weights and generated plots
}

# ==========================================
# Reproducibility and directory setup
# ==========================================
random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
torch.use_deterministic_algorithms(False)
os.makedirs(CONFIG["save_dir"], exist_ok=True)
torch.backends.cudnn.benchmark = True # Optimizes performance if input sizes remain constant

# ==========================================
# FID Calculation & Inception Feature Extractor
# ==========================================
class InceptionFeatureExtractor(nn.Module):
    """Extracts 2048-dimensional feature vectors using a pre-trained Inception V3 network."""
    def __init__(self, device):
        super(InceptionFeatureExtractor, self).__init__()
        self.device = device
        
        # Load pre-trained Inception v3
        # The pre-trained weights require aux_logits=True during initialization.
        self.model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
        
        # Replace the final classification layer with an Identity layer 
        # so we get the 2048-dimensional pooled features instead of 1000 class logits
        self.model.fc = nn.Identity()
        
        # Setting to eval() mode automatically turns off the aux_logits output 
        # during the forward pass, returning just our desired 2048-d tensor.
        self.model.eval()
        self.model.to(self.device)

    def forward(self, x):
        # The generated images and CelebA images are in the range [-1, 1] due to our Normalize/Tanh.
        # PyTorch's InceptionV3 transform_input=True expects data in the range [0, 1].
        # We manually shift it to [0, 1] here before processing.
        x = (x + 1.0) / 2.0
        
        # Inception V3 was trained on 299x299 images, so we must interpolate
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            features = self.model(x)
        return features

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance math formula."""
    diff = mu1 - mu2
    
    # Calculate the square root of the dot product of covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Handle singular product errors by adding an epsilon offset
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Remove imaginary components caused by numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    # FID Formula: ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def calculate_fid(netG, dataloader, feature_extractor, config, device, num_samples=1000):
    """Generates fake images, extracts features for real and fake, and computes FID."""
    print(f"\nCalculating FID over {num_samples} samples...")
    netG.eval() # Set generator to evaluation mode
    
    real_features = []
    fake_features = []
    collected = 0
    
    for data in dataloader:
        real_imgs = data[0].to(device)
        b_size = real_imgs.size(0)
        
        # Cut off exactly at num_samples
        if collected + b_size > num_samples:
            real_imgs = real_imgs[:num_samples - collected]
            b_size = real_imgs.size(0)
            
        # 1. Extract Real Features
        with torch.no_grad():
            real_feat = feature_extractor(real_imgs)
        real_features.append(real_feat.cpu().numpy())
        
        # 2. Generate and Extract Fake Features
        noise = torch.randn(b_size, config["nz"], 1, 1, device=device)
        with torch.no_grad():
            fake_imgs = netG(noise)
            fake_feat = feature_extractor(fake_imgs)
        fake_features.append(fake_feat.cpu().numpy())
        
        collected += b_size
        if collected >= num_samples:
            break
            
    netG.train() # Switch back to training mode
    
    # Concatenate all batches 
    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)
    
    # Calculate Mean and Covariance for real and fake distributions
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Compute Final Score
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
        
        nz = CONFIG["nz"]
        ngf = CONFIG["ngf"]
        nc = CONFIG["nc"]
        
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
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

class Discriminator(nn.Module):
    """
    Standard DCGAN Discriminator modified with Spectral Normalization.
    Spectral Normalization stabilizes training by constraining the Lipschitz constant of the network,
    acting as a powerful alternative to Gradient Penalty.
    """
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        nc, ndf = CONFIG["nc"], CONFIG["ndf"]
        
        self.main = nn.Sequential(
            # Notice we wrap nn.Conv2d in spectral_norm() and removed BatchNorm2d
            # Input size: (nc) x 64 x 64
            spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf) x 32 x 32
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf*2) x 16 x 16
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf*4) x 8 x 8
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # --- FIX 1: Removed spectral_norm on the final layer ---
            # Output layer mapping down to 1 channel, passing through Sigmoid for probability (0 to 1)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Instantiate the Discriminator and apply weight initialization
netD = Discriminator(CONFIG["ngpu"]).to(device)
netD.apply(weights_init)


# ==========================================
# 3. Training Loop 
# ==========================================
if __name__ == '__main__':
    # Binary Cross Entropy Loss function
    criterion = nn.BCELoss()
    
    # Create a fixed noise vector to visually track Generator's progress across epochs
    fixed_noise = torch.randn(64, CONFIG["nz"], 1, 1, device=device)
    
    # --- FIX 2: Turned off Label Smoothing ---
    # We use strict 1.0 for real and 0.0 for fake labels
    real_label, fake_label = 1.0, 0.0

    # Custom learning rates for Generator and Discriminator (Two-Time Scale Update Rule - TTUR)
    lrd = 0.0003
    lrg = 0.0001    
    
    # Setup Adam optimizers
    optimizerD = optim.Adam(netD.parameters(), lrd, betas=(CONFIG["beta1"], 0.999))
    optimizerG = optim.Adam(netG.parameters(), lrg, betas=(CONFIG["beta1"], 0.999))
    
    # Initialize the Feature Extractor for FID evaluation
    feature_extractor = InceptionFeatureExtractor(device)

    # Trackers for plots and logging
    img_list = []
    G_losses = []  
    D_losses = []  
    d_epoch_losses = []  
    g_epoch_losses = []  
    fid_scores = []     

    iters = 0

    print("Starting Training Loop...")
    for epoch in range(CONFIG["num_epochs"]):
        running_d_loss = 0.0
        running_g_loss = 0.0
        
        for i, data in enumerate(dataloader, 0):
            # ==========================================
            # 1. Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # ==========================================
            netD.zero_grad()
            
            # Format batch of real images
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            
            # Add occasional noise to labels (flips 5% of real labels to fake) to prevent D from overpowering G
            if random.random() < 0.05:
                label.fill_(fake_label)
            
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label) # Loss on real images
            errD_real.backward()
            D_x = output.mean().item() # Average prediction on real batch

            # Generate batch of fake images
            noise = torch.randn(b_size, CONFIG["nz"], 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label) # Labels are 0 for fakes
            
            # Forward pass fake batch through D (detach to avoid computing gradients for G)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label) # Loss on fake images
            errD_fake.backward()
            D_G_z1 = output.mean().item() # Average prediction on fake batch before G update
            
            # Add the gradients from the all-real and all-fake batches, then update D
            errD = errD_real + errD_fake
            optimizerD.step()

            # ==========================================
            # 2. Update G network: maximize log(D(G(z)))
            # ==========================================
            netG.zero_grad()
            label.fill_(real_label)  # Fake labels are treated as real (1) for generator cost
            
            # Forward pass fake batch through D (again, but attached to the graph this time)
            output = netD(fake).view(-1)
            errG = criterion(output, label) # Loss on G fooling D
            errG.backward()
            D_G_z2 = output.mean().item() # Average prediction on fake batch after G update
            optimizerG.step()

            # Accumulate loss for epoch averages
            running_d_loss += errD.item()
            running_g_loss += errG.item()

            # Output training stats periodically
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, CONFIG["num_epochs"], i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Save generated images occasionally and at the very end of the epoch
            if (iters % 500 == 0) or ((epoch == CONFIG["num_epochs"]-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake_grid = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake_grid, padding=2, normalize=True))

            iters += 1

        # Store average losses for this epoch
        d_epoch_losses.append(running_d_loss / len(dataloader))
        g_epoch_losses.append(running_g_loss / len(dataloader))

        # Calculate FID at the end of the epoch (sample 1000 images for speed)
        current_fid = calculate_fid(netG, dataloader, feature_extractor, CONFIG, device, num_samples=1000)
        fid_scores.append(current_fid)

        # Save model weights (checkpoints) at the end of each epoch
        torch.save(netG.state_dict(), os.path.join(CONFIG["save_dir"], f"netG_epoch_{epoch}.pth"))
        torch.save(netD.state_dict(), os.path.join(CONFIG["save_dir"], f"netD_epoch_{epoch}.pth"))
        
        print(f"--- Epoch [{epoch+1}/{CONFIG['num_epochs']}] Summary ---")
        print(f"Avg D Loss: {d_epoch_losses[-1]:.4f} | Avg G Loss: {g_epoch_losses[-1]:.4f} | FID: {current_fid:.4f}")
        print(f"Checkpoints saved for epoch {epoch} in {CONFIG['save_dir']}\n")

    # ==========================================
    # 4. Post-Training: Save, Graph, and Visualize
    # ==========================================
    print("Training Complete. Saving metrics and generating plots...")
    
    # Save the raw data arrays to a NumPy .npz file so they can be analyzed later without retraining
    metrics_path = os.path.join(CONFIG["save_dir"], "training_metrics.npz")
    np.savez(metrics_path, G_losses=G_losses, D_losses=D_losses, fid_scores=fid_scores)
    print(f"Metrics saved successfully to {metrics_path}")

    # Plot 1: Generator and Discriminator Losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G Loss", alpha=0.7)
    plt.plot(D_losses, label="D Loss", alpha=0.7)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    loss_plot_path = os.path.join(CONFIG["save_dir"], "loss_plot.png")
    plt.savefig(loss_plot_path)
    print(f"Saved loss plot to {loss_plot_path}")
    plt.show()

    # Plot 2: FID Scores Over Epochs
    plt.figure(figsize=(10, 5))
    plt.title("FID Score Over Epochs")
    # X-axis is 1 to num_epochs
    epochs_range = range(1, CONFIG["num_epochs"] + 1)
    plt.plot(epochs_range, fid_scores, marker='o', color='red', label="FID")
    plt.xlabel("Epochs")
    plt.ylabel("FID Score")
    plt.xticks(epochs_range)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    fid_plot_path = os.path.join(CONFIG["save_dir"], "fid_plot.png")
    plt.savefig(fid_plot_path)
    print(f"Saved FID plot to {fid_plot_path}")
    plt.show()

    # Plot 3: Display Generated Photos
    print("Displaying final generated images...")
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated Images at Final Epoch")
    
    # We transpose the image tensor from (C, H, W) to (H, W, C) so matplotlib can read it
    final_images = np.transpose(img_list[-1], (1, 2, 0))
    plt.imshow(final_images)
    
    final_img_path = os.path.join(CONFIG["save_dir"], "final_generated_images.png")
    plt.savefig(final_img_path)
    print(f"Saved final generated images to {final_img_path}")
    plt.show()

    # ==========================================
    # Display Total Execution Time
    # ==========================================
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate hours, minutes, and seconds
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    # Convert to integers safely before formatting
    h = int(hours)
    m = int(minutes)
    
    print("\n==========================================")
    # Using standard .format() to prevent editor/linter warnings
    print("Total Execution Time: {:02d}h {:02d}m {:05.2f}s".format(h, m, seconds))
    print("==========================================")