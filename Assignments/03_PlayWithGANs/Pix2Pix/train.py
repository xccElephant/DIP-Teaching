import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from discriminator import Discriminator
from generator import Generator
from pix2pix_dataset import Pix2PixDataset
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image


def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f"{folder_name}/epoch_{epoch}", exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f"{folder_name}/epoch_{epoch}/result_{i + 1}.png", comparison)


def train_one_epoch(
    generator, discriminator, dataloader, g_optimizer, d_optimizer, 
    criterion_GAN, criterion_pixelwise, device, epoch, num_epochs, scaler,
    dataset_name, lambda_pixel=100
):
    """
    Train the model for one epoch.

    Args:
        generator (nn.Module): The generator model.
        discriminator (nn.Module): The discriminator model.
        dataloader (DataLoader): DataLoader for the training data.
        g_optimizer (Optimizer): Optimizer for the generator.
        d_optimizer (Optimizer): Optimizer for the discriminator.
        criterion_GAN (Loss): Loss function for the GAN.
        criterion_pixelwise (Loss): Loss function for the pixelwise loss.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
        scaler (GradScaler): GradScaler for mixed precision training.
        dataset_name (str): Name of the dataset.
        lambda_pixel (float): Weight for the pixelwise loss.
    """
    generator.train()
    discriminator.train()
    total_g_loss = 0.0
    total_d_loss = 0.0

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        batch_size = image_rgb.size(0)
        real = torch.ones((batch_size, 1, 15, 15), device=device)
        fake = torch.zeros((batch_size, 1, 15, 15), device=device)
        
        # Move data to the device
        image_rgb = image_rgb.to(device, non_blocking=True)
        image_semantic = image_semantic.to(device, non_blocking=True)

        # -----------------
        # train discriminator
        # -----------------
        d_optimizer.zero_grad()

        # Forward pass
        with torch.amp.autocast("cuda"):
            # generate fake image
            fake_semantic = generator(image_rgb)
            
            # discriminator's real image result
            real_loss = criterion_GAN(discriminator(image_rgb, image_semantic), real)
            # 生成图像的判别结果
            fake_loss = criterion_GAN(discriminator(image_rgb, fake_semantic.detach()), fake)
            
            d_loss = (real_loss + fake_loss) / 2

        # Backward pass and optimization
        scaler.scale(d_loss).backward()
        scaler.step(d_optimizer)

        # -----------------
        # train generator
        # -----------------
        g_optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            # GAN loss
            g_loss_gan = criterion_GAN(discriminator(image_rgb, fake_semantic), real)
            # Pixel-wise loss
            g_loss_pixel = criterion_pixelwise(fake_semantic, image_semantic)
            
            # Total generator loss
            g_loss = g_loss_gan + lambda_pixel * g_loss_pixel

        scaler.scale(g_loss).backward()
        scaler.step(g_optimizer)
        scaler.update()

        # Print loss information
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], "
            f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}"
        )

        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()

        # Save sample images every 5 epochs
        if epoch % 5 == 0 and i == 0:
            save_images(
                image_rgb,
                image_semantic,
                outputs,
                os.path.join("train_results", f"{dataset_name}"),
                epoch,
            )

    return total_g_loss / len(dataloader), total_d_loss / len(dataloader)


def validate(model, dataloader, criterion, device, epoch, num_epochs, dataset_name):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Forward pass
            outputs = model(image_rgb)

            # Compute the loss
            loss = criterion(outputs, image_semantic)
            val_loss += loss.item()

            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                save_images(
                    image_rgb,
                    image_semantic,
                    outputs,
                    os.path.join("val_results", f"{dataset_name}"),
                    epoch,
                )

    # Calculate average validation loss
    avg_val_loss = val_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

    return avg_val_loss


def main():
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize datasets and dataloaders
    train_dataset = Pix2PixDataset(list_file="train_list.txt")
    val_dataset = Pix2PixDataset(list_file="val_list.txt")

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True
    )

    scaler = torch.amp.GradScaler("cuda")

    # Initialize generator and discriminator
    generator = FullyConvNetwork().to(device)
    discriminator = Discriminator().to(device)

    # Loss function
    criterion_GAN = nn.BCELoss()
    criterion_pixelwise = nn.L1Loss()

    # Optimizer
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    num_epochs = 200

    # Add a learning rate scheduler for decay
    scheduler = StepLR(optimizer, step_size=num_epochs // 10, gamma=0.2)

    dataset_name = "cityscapes"

    loss_file = open(os.path.join("logs", f"loss_history_{dataset_name}.txt"), "w")
    loss_file.write("Epoch\tTrain Loss\tValidation Loss\n")

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            num_epochs,
            scaler,
            dataset_name,
        )
        val_loss = validate(
            model, val_loader, criterion, device, epoch, num_epochs, dataset_name
        )

        loss_file.write(f"{epoch + 1}\t{train_loss:.4f}\t{val_loss:.4f}\n")
        loss_file.flush()

        # Step the scheduler after each epoch
        scheduler.step()

        # Save model checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            os.makedirs(os.path.join("checkpoints", f"{dataset_name}"), exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(
                    "checkpoints",
                    f"{dataset_name}",
                    f"pix2pix_model_epoch_{epoch + 1}.pth",
                ),
            )

    loss_file.close()


if __name__ == "__main__":
    main()