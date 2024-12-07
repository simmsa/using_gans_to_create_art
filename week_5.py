import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

# --- Dataset Preparation ---
data_dir = Path("./data").resolve()

# Define Transformations
transform = transforms.Compose(
    [
        # transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
    ]
)


# Define Custom Dataset
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.image_paths = list(
            self.root_dir.glob("*.jpg")
        )  # Adjust for other formats if needed
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Read the image with PIL
        image = Image.open(img_path).convert("RGB")  # Ensure it's in RGB format
        if self.transform:
            image = self.transform(image)
        return image, 0  # Return image and a dummy label


# Function to compute the mean and std for Monet images
def compute_mean_std(image_paths):
    means = np.zeros(3)
    stds = np.zeros(3)
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img = np.array(img) / 255.0  # Normalize to [0, 1]
        means += img.mean(axis=(0, 1))  # Mean per channel (RGB)
        stds += img.std(axis=(0, 1))  # Standard deviation per channel (RGB)

    means /= len(image_paths)
    stds /= len(image_paths)

    return means, stds


# Paths to the Monet images
monet_image_paths = list(Path("./data/monet_jpg").glob("*.jpg"))
monet_means, monet_stds = compute_mean_std(monet_image_paths)

photo_image_paths = list(Path("./data/photo_jpg").glob("*.jpg"))
photo_means, photo_stds = compute_mean_std(photo_image_paths)

print(monet_means, monet_stds)
print(photo_means, photo_stds)


# Define the complete transform with the custom Monet-style normalization
monet_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        # transforms.Normalize(mean=monet_means, std=monet_stds),
    ]
)
photo_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        # transforms.Normalize(mean=photo_means, std=photo_stds),
    ]
)


photo_dataset = ImageDataset(
    root_dir=Path(data_dir, "photo_jpg"), transform=photo_transform
)
monet_dataset = ImageDataset(
    root_dir=Path(data_dir, "monet_jpg"), transform=monet_transform
)

photo_loader = DataLoader(photo_dataset, batch_size=1, shuffle=False)
monet_loader = DataLoader(monet_dataset, batch_size=1, shuffle=False)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels, channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_residual_blocks=9):
        super(Generator, self).__init__()
        model = [
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        in_features = 64
        for _ in range(2):
            out_features = in_features * 2
            model += [
                nn.Conv2d(
                    in_features,
                    out_features,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual Blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        for _ in range(2):
            out_features = in_features // 2
            model += [
                nn.ConvTranspose2d(
                    in_features,
                    out_features,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output Layer
        model += [
            nn.Conv2d(in_features, out_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride):
            return [
                nn.Conv2d(
                    in_filters,
                    out_filters,
                    kernel_size=4,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(out_filters),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, stride=2),
            *discriminator_block(64, 128, stride=2),
            *discriminator_block(128, 256, stride=2),
            *discriminator_block(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)


# Initialize Models
G = Generator().to(device)  # Photo -> Monet
F = Generator().to(device)  # Monet -> Photo
D_M = Discriminator().to(device)  # Monet Discriminator
D_P = Discriminator().to(device)  # Photo Discriminator

# Optimizers
lr = 0.0002
# lr = 0.001
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
F_optimizer = optim.Adam(F.parameters(), lr=lr, betas=(0.5, 0.999))
D_M_optimizer = optim.Adam(D_M.parameters(), lr=lr, betas=(0.5, 0.999))
D_P_optimizer = optim.Adam(D_P.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss Functions
adversarial_loss = nn.MSELoss()
cycle_loss = nn.L1Loss()
identity_loss = nn.L1Loss()

# --- Training Loop ---
epochs = 250


# List to store statistics for each epoch
training_stats = []

for epoch in range(1, epochs + 1):
    start_time = time.time()

    # Dictionary to track cumulative stats for the current epoch
    epoch_stats = {
        "Epoch": epoch,  # Current epoch number
        "Generator Total Loss": 0,  # Total loss for both generators (G and F)
        "Monet Discriminator Loss": 0,  # Loss for Monet discriminator (D_M)
        "Photo Discriminator Loss": 0,  # Loss for Photo discriminator (D_P)
        "Cycle Consistency Loss Photo": 0,  # Cycle loss for photo reconstructions
        "Cycle Consistency Loss Monet": 0,  # Cycle loss for Monet reconstructions
        "Identity Loss Photo": 0,  # Identity loss for photo images
        "Identity Loss Monet": 0,  # Identity loss for Monet images
        "Monet Discriminator Accuracy": 0,  # Accuracy of Monet discriminator
        "Photo Discriminator Accuracy": 0,  # Accuracy of Photo discriminator
    }

    for i, (photo_batch, monet_batch) in enumerate(zip(photo_loader, monet_loader)):
        real_photo = photo_batch[0].to(device)
        # print(real_photo.shape)
        real_monet = monet_batch[0].to(device)
        # print(real_monet.shape)

        # ------------------------------------
        # Step 1: Update Generators (G and F)
        # ------------------------------------
        G_optimizer.zero_grad()
        F_optimizer.zero_grad()

        # Photo -> Monet -> Photo (Cycle 1)
        fake_monet = G(real_photo)
        reconstructed_photo = F(fake_monet)

        # print(fake_monet.shape)
        # print(reconstructed_photo.shape)

        # Monet -> Photo -> Monet (Cycle 2)
        fake_photo = F(real_monet)
        reconstructed_monet = G(fake_photo)

        # print(fake_photo.shape)
        # print(reconstructed_monet.shape)

        # Adversarial Loss
        valid = torch.ones_like(D_M(fake_monet))
        g_loss_monet = adversarial_loss(D_M(fake_monet), valid)

        valid = torch.ones_like(D_P(fake_photo))
        g_loss_photo = adversarial_loss(D_P(fake_photo), valid)

        # Cycle Consistency Loss
        cycle_loss_photo = cycle_loss(reconstructed_photo, real_photo)
        cycle_loss_monet = cycle_loss(reconstructed_monet, real_monet)

        # Identity Loss
        identity_photo = identity_loss(F(real_photo), real_photo)
        identity_monet = identity_loss(G(real_monet), real_monet)

        # Total Generator Loss
        total_g_loss = (
            g_loss_monet
            + g_loss_photo
            + 10 * (cycle_loss_photo + cycle_loss_monet)
            + 5 * (identity_photo + identity_monet)
        )
        total_g_loss.backward()
        G_optimizer.step()
        F_optimizer.step()

        # ------------------------------------
        # Step 2: Update Discriminators
        # ------------------------------------
        D_M_optimizer.zero_grad()
        D_P_optimizer.zero_grad()

        # Real Loss for Monet
        real_validity = torch.ones_like(D_M(real_monet))
        real_loss = adversarial_loss(D_M(real_monet), real_validity)

        # Fake Loss for Monet
        fake_validity = torch.zeros_like(D_M(fake_monet.detach()))
        fake_loss = adversarial_loss(D_M(fake_monet.detach()), fake_validity)

        # Total Discriminator Loss for Monet
        d_m_loss = (real_loss + fake_loss) / 2
        d_m_loss.backward()
        D_M_optimizer.step()

        # Accuracy for Monet Discriminator
        d_m_predictions = torch.cat([real_validity, fake_validity]).cpu().numpy()
        d_m_labels = (
            torch.cat([torch.ones_like(real_validity), torch.zeros_like(fake_validity)])
            .cpu()
            .numpy()
        )

        # print(d_m_labels.flatten())
        # print(d_m_predictions.flatten())
        # print(type(d_m_labels.flatten()))
        # print(type(d_m_predictions.flatten()))

        d_m_accuracy = accuracy_score(
            # d_m_labels.flatten(), d_m_predictions.round().flatten()
            d_m_labels.flatten(),
            d_m_predictions.flatten(),
        )

        # Real Loss for Photos
        real_validity = torch.ones_like(D_P(real_photo))
        real_loss = adversarial_loss(D_P(real_photo), real_validity)

        # Fake Loss for Photos
        fake_validity = torch.zeros_like(D_P(fake_photo.detach()))
        fake_loss = adversarial_loss(D_P(fake_photo.detach()), fake_validity)

        # Total Discriminator Loss for Photos
        d_p_loss = (real_loss + fake_loss) / 2
        d_p_loss.backward()
        D_P_optimizer.step()

        # Accuracy for Photo Discriminator
        d_p_predictions = torch.cat([real_validity, fake_validity]).cpu().numpy()
        d_p_labels = (
            torch.cat([torch.ones_like(real_validity), torch.zeros_like(fake_validity)])
            .cpu()
            .numpy()
        )
        d_p_accuracy = accuracy_score(d_p_labels.flatten(), d_p_predictions.flatten())

        # ------------------------------------
        # Aggregate Metrics
        # ------------------------------------
        epoch_stats["Generator Total Loss"] += total_g_loss.item()
        epoch_stats["Monet Discriminator Loss"] += d_m_loss.item()
        epoch_stats["Photo Discriminator Loss"] += d_p_loss.item()
        epoch_stats["Cycle Consistency Loss Photo"] += cycle_loss_photo.item()
        epoch_stats["Cycle Consistency Loss Monet"] += cycle_loss_monet.item()
        epoch_stats["Identity Loss Photo"] += identity_photo.item()
        epoch_stats["Identity Loss Monet"] += identity_monet.item()
        epoch_stats["Monet Discriminator Accuracy"] += d_m_accuracy
        epoch_stats["Photo Discriminator Accuracy"] += d_p_accuracy

        if i % 100 == 0:
            print(
                f"Epoch [{epoch}/{epochs}], Step [{i}], "
                f"Generator Loss: {total_g_loss.item():.4f}, "
                f"D_M Loss: {d_m_loss.item():.4f}, D_M Accuracy: {d_m_accuracy:.2%}, "
                f"{d_p_accuracy:.2%}"
            )

    # Normalize the accumulated stats by the number of steps
    num_batches = len(photo_loader)
    epoch_stats["Generator Total Loss"] /= num_batches
    epoch_stats["Monet Discriminator Loss"] /= num_batches
    epoch_stats["Photo Discriminator Loss"] /= num_batches
    epoch_stats["Cycle Consistency Loss Photo"] /= num_batches
    epoch_stats["Cycle Consistency Loss Monet"] /= num_batches
    epoch_stats["Identity Loss Photo"] /= num_batches
    epoch_stats["Identity Loss Monet"] /= num_batches
    epoch_stats["Monet Discriminator Accuracy"] /= num_batches
    epoch_stats["Photo Discriminator Accuracy"] /= num_batches

    # Calculate the total loss for the epoch
    epoch_stats["Total Loss"] = (
        epoch_stats["Generator Total Loss"]
        + epoch_stats["Monet Discriminator Loss"]
        + epoch_stats["Photo Discriminator Loss"]
    )

    # Add elapsed time for the epoch
    epoch_stats["Epoch Time (s)"] = time.time() - start_time
    training_stats.append(epoch_stats)

    should_evaluate = False

    if epoch <= 100:
        if epoch % 25 == 0:
            should_evaluate = True
    elif epoch <= 150:
        if epoch % 10 == 0:
            should_evaluate = True
    elif epoch <= 250:
        if epoch % 5 == 0:
            should_evaluate = True

    # --- Evaluation Every 5 Epochs ---
    if should_evaluate is True:
        print(f"Evaluation at epoch {epoch}")

        # Save the model or generate images for evaluation purposes
        # For example, generating output images for the Monet-style images
        output_dir = f"./output_images/no_transform.smaller_kernel.{epoch}_epochs"
        os.makedirs(output_dir, exist_ok=True)

        G.eval()
        with torch.no_grad():
            for i, (photo, _) in enumerate(photo_loader):
                photo = photo.to(device)
                fake_monet = G(photo)
                fake_monet = (fake_monet + 1) / 2  # Denormalize to [0, 1]
                fake_monet = transforms.ToPILImage()(fake_monet.squeeze().cpu())
                fake_monet.save(os.path.join(output_dir, f"image_{i + 1:04d}.jpg"))
        # --- Create Submission Zip ---
        with zipfile.ZipFile(
            f"./output_zip/no_transform.smaller_kernel.{epoch}_epoch_images.zip", "w"
        ) as zf:
            for file in Path(output_dir).glob("*.jpg"):
                zf.write(file, arcname=file.name)

    # Convert the collected stats into a DataFrame and save to parquet
    training_df = pd.DataFrame(training_stats)
    training_df.to_parquet(
        f"./training_stats/training_stats.no_transform.{epoch}_of_{epochs}_epochs.parquet"
    )

    # Print a summary of the final stats
