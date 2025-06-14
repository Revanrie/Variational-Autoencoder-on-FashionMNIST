#!/usr/bin/env python
# coding: utf-8

# # Variational Autoencoder on FashionMNIST
# **Author:** Yusuf Adamu  
# **Objective:** Build, train, and analyze a Variational Autoencoder (VAE) from scratch using PyTorch on the FashionMNIST dataset.  
# **PyTorch, deep learning, generative modeling, latent space visualization, custom loss functions.**
# 
# 
# 

# ##  Introduction
# 
# Variational Autoencoders (VAEs) are powerful generative models that learn a latent representation of data in an unsupervised manner. In this project, I implement a VAE from scratch using PyTorch and train it on the FashionMNIST dataset, which contains grayscale images of clothing items.
# 
# The key goals of this notebook are to:
# 
# - Build a fully functional encoder-decoder architecture using neural networks.
# - Implement the **reparameterization trick** to enable backpropagation through stochastic layers.
# - Define and optimize a **custom VAE loss** combining reconstruction loss and KL divergence.
# - Visualize the learned **2D latent space** to gain insight into how the model separates different clothing categories.
# - Generate **new images** by decoding samples from different regions of the latent space.
# 

# In[ ]:





# Import useful libraries

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# Load the FashionMNIST data

# In[ ]:


#Load amd Normalize the FashionMNIST Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)


# Visualize the original data to have an idea of the images and its categories
# 

# In[ ]:


#Visualize a small subset original Datasets
# define a function to show an image
def imshow(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()

# Get some of the training images
data_iter = iter(train_loader)
images, labels = next(data_iter)
# Show the images
imshow(torchvision.utils.make_grid(images))
print(labels[:10])


# In[ ]:


# Define the Encoder Network
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc_mean = nn.Linear(512, 2)
        self.fc_log_var = nn.Linear(512, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        return mean, log_var


# In[ ]:


# Define the Decoder Network
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(2, 512)
        self.fc2 = nn.Linear(512, 784)

    def forward(self, z):
        z = torch.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(z))


# In[ ]:


# Define Reparameterization Trick
def reparameterize(mean, log_var):
    std = torch.exp(0.5 * log_var)
    epsilon = torch.randn_like(std)
    return mean + epsilon * std


# In[ ]:


# Define the VAE Loss Function (defined by Reconstruction Loss + KL Divergence)
def loss_function(reconstructed_x, x, mean, log_var):
    reconstruction_loss = nn.functional.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + kl_divergence


# In[ ]:


#Initialize the Encoder, Decoder, and Optimizer
encoder = Encoder()
decoder = Decoder()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)


# Train the Variational Autoencoder (VAE)

# In[ ]:


# Training the VAE

#to store loss per epoc lets use a list
train_loss_values = []

def train(epoch):
    encoder.train()
    decoder.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, 784)
        optimizer.zero_grad()
        mean, log_var = encoder(data)
        z = reparameterize(mean, log_var)
        reconstructed_x = decoder(z)
        loss = loss_function(reconstructed_x, data, mean, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    average_loss = train_loss / len(train_loader.dataset)
    train_loss_values.append(average_loss)
    print(f'Epoch {epoch}, Loss: {average_loss}')

for epoch in range(1, 11):
    train(epoch)


#  Training curves (loss vs epochs)

# In[ ]:


# Plot Training Loss Curve
plt.plot(train_loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()


# 2D-scatter plot of the Z values corresponding to a random subset of 5000
# FashionMNIST input images

# In[ ]:


# Step 9: Latent Space Visualization
z_points = []
encoder.eval()
with torch.no_grad():
    for i in range(5000):
        img, _ = train_data[i]
        img = img.view(-1, 784)
        mean, log_var = encoder(img)
        z = reparameterize(mean, log_var)
        z_points.append(z.numpy())

z_points = np.concatenate(z_points, axis=0)
plt.scatter(z_points[:, 0], z_points[:, 1], alpha=0.5)
plt.title('Latent Space')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.show()


# In[ ]:


# Step 9: Latent Space Visualization with Category Coloring
def plot_latent_space_with_colors(encoder, train_data, num_samples=5000):
    z_points = []
    labels = []
    encoder.eval()

    with torch.no_grad():
        for i in range(num_samples):
            img, label = train_data[i]
            # Flatten the image
            img = img.view(-1, 784)
            mean, log_var = encoder(img)
            z = reparameterize(mean, log_var)

            # Remove the extra dimensions
            z_points.append(z.squeeze(0).numpy())
            #Append the corresponding labels
            labels.append(label)

    z_points = np.array(z_points)
    labels = np.array(labels)

    print(f"Shape of z_points after squeezing: {z_points.shape}")

    # Plotting the 2-D scatter plot with different colors for each category
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(z_points[:, 0], z_points[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(10), label='Category')
    plt.title('2D Latent Space of 5000 Random FashionMNIST Images')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.show()

plot_latent_space_with_colors(encoder, train_data)


# Grid plot of 15x15 generated images which are the outputs of the decoder
# generated for the complete range of z values across the two components of z

# In[ ]:


# Lets Get 15x15 Grid of Images from the Decoder
with torch.no_grad():
    grid_x = np.linspace(-3, 3, 15)
    grid_y = np.linspace(-3, 3, 15)
    figure = np.zeros((28 * 15, 28 * 15))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = torch.tensor([[xi, yi]]).float()
            x_decoded = decoder(z_sample).view(28, 28)
            figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = x_decoded.numpy()

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='gray')
plt.show()


# Grid of generated images where each row/column corresponds to
# varying images for one category

# In[ ]:


# Generate Varying Images for each Category

# define a function to collect latent vectors for each category
def collect_latent_vectors_per_category():
    encoder.eval()
    category_latents = {i: [] for i in range(10)}

    with torch.no_grad():
        for data, label in train_loader:
            data = data.view(-1, 784)
            mean, log_var = encoder(data)
            z = reparameterize(mean, log_var)

            for i in range(len(label)):
                category_latents[label[i].item()].append(z[i].cpu().numpy())

    # Compute the mean latent vector for each category
    category_latent_means = {i: np.mean(category_latents[i], axis=0) for i in category_latents}

    return category_latent_means


#Collect latent vectors and calculate mean z for each category
category_latent_means = collect_latent_vectors_per_category()

# lets Generate and plot a grid of images for each category by varying z
def generate_images_for_each_category(category_latent_means):
    num_categories = 10
    num_variations = 5

    plt.figure(figsize=(10, 10))

    for i in range(num_categories):
        mean_z = category_latent_means[i]

        #Generate images by varying the mean latent vector
        for j in range(num_variations):
            # lets add random noise to get variations
            z_varied = torch.tensor(mean_z + np.random.normal(0, 0.5, size=mean_z.shape)).float().unsqueeze(0)

            # Decode
            with torch.no_grad():
                generated_image = decoder(z_varied).view(28, 28).cpu().numpy()

            # lets Plot the image in the corresponding grid position
            ax = plt.subplot(num_categories, num_variations, i * num_variations + j + 1)
            ax.axis('off')
            plt.imshow(generated_image, cmap='gray')

    plt.show()

#Generate and visualize the images
generate_images_for_each_category(category_latent_means)


# ## 🧾 Conclusion
# 
# This notebook explored the implementation of a Variational Autoencoder (VAE) on the FashionMNIST dataset using PyTorch. Through this work, I investigated how VAEs can effectively learn low-dimensional latent representations of high-dimensional image data, and how those representations can be visualized and sampled to generate new data.
# 
# Key outcomes of the project include:
# 
# - Successful training of a VAE with stable convergence
# - Visualization of meaningful 2D latent space clusters across FashionMNIST categories
# - Demonstration of controlled image generation using both grid sampling and class-averaged latent vectors
# 
# This project provided an opportunity to apply core ideas in variational inference, generative modeling, and neural network design. It also served as a hands-on framework to explore how abstract latent spaces can capture structural information in image data.
# 
# Further extensions could involve increasing the latent space dimensionality, incorporating convolutional layers for better reconstructions, or experimenting with conditional VAE structures.
# 

# In[ ]:




