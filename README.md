# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

Problem: 

To develop and train a Denoising Autoencoder using PyTorch for removing noise from handwritten digit images in the MNIST dataset and reconstructing clean images accurately.

Dataset:

The MNIST dataset is a collection of grayscale handwritten digit images ranging from 0 to 9. Each image is of size 28 × 28 pixels. The dataset contains:

Training Images: 60,000

Testing Images: 10,000

Number of Classes: 10

Image Type: Grayscale

Image Size: 28 × 28 pixels

The dataset is loaded directly using torchvision.datasets.MNIST in PyTorch.

## DESIGN STEPS
### STEP 1: 

Import the required libraries and load the MNIST dataset using torchvision.datasets. Apply transformations such as converting images into tensors.

### STEP 2: 

Create training and testing datasets and use DataLoader for batch processing during model training and evaluation.

### STEP 3: 

Add random noise to the input images using a custom noise function to create noisy image samples.


### STEP 4: 

Build the Denoising Autoencoder model using convolutional layers for the encoder and transposed convolutional layers for the decoder.

### STEP 5: 

Initialize the autoencoder model, define the loss function (MSELoss), and configure the optimizer (Adam).

### STEP 6: 

Train the autoencoder model using noisy images as input and original clean images as target outputs for multiple epochs.

### STEP 7:

Evaluate the trained autoencoder model using test images and generate denoised outputs from noisy input images.

### STEP 8:

Visualize the original images, noisy images, and denoised reconstructed images using Matplotlib for performance comparison.

## PROGRAM

### Name: Amruthavarshini Gopal

### Register Number: 212223230013

```python
# Autoencoder Definition
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
      super(DenoisingAutoencoder,self).__init__()
      self.encoder=nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,kernel_size=3,stride=2,padding=1),
            nn.ReLU()
        )
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(32,16,kernel_size=3,stride=2,output_padding=1,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,kernel_size=3,stride=2,output_padding=1,padding=1),
            nn.Sigmoid()
        )
     def forward(self, x):
        # Include your code here
        x=self.encoder(x)
        x=self.decoder(x)
        return x

# Initialize model
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=1e-3)

# Train the autoencoder
def train(model, loader, criterion, optimizer, epochs=5):
    # Include your code here
    model.train()
    print("Name: Amruthavarshini Gopal")
    print("Register Number: 212223230013")
    for epoch in range(epochs):
      running_loss=0.0
      for images, _ in loader:
        images=images.to(device)
        noisy_images=add_noise(images).to(device)

        outputs=model(noisy_images)
        loss=criterion(outputs,images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
      print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")

# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name: Amruthavarshini Gopal")
    print("Register Number: 212223230013")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
# Run training and visualization
train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)

```

## OUTPUT

### Model Summary
<img width="963" height="563" alt="Screenshot 2026-05-22 114504" src="https://github.com/user-attachments/assets/a6d2e00c-f38b-4d8e-96da-8b636703ebe5" />


### Training loss
<img width="449" height="179" alt="Screenshot 2026-05-22 114603" src="https://github.com/user-attachments/assets/fc219f20-2509-4965-af6f-c5a31187d45c" />

## Original vs Noisy Vs Reconstructed Image
<img width="1711" height="581" alt="Screenshot 2026-05-22 114618" src="https://github.com/user-attachments/assets/906e0074-11ca-4711-aef7-1e17b33cfb7b" />


## RESULT
Thus, a Denoising Autoencoder model was successfully developed and trained using PyTorch for image denoising on the MNIST dataset. The model effectively reconstructed clean images from noisy inputs, and the denoising performance was visualized successfully using original, noisy, and reconstructed images.
