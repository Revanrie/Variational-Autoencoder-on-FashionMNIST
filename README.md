# Variational Autoencoder on FashionMNIST 🧵👗

![VAE on FashionMNIST](https://img.shields.io/badge/Download%20Releases-Here-blue?style=for-the-badge&logo=github&logoColor=white)

Welcome to the **Variational Autoencoder on FashionMNIST** repository! This project focuses on building and visualizing a 2D latent space using a custom Variational Autoencoder (VAE) trained on the FashionMNIST dataset. With the power of PyTorch, you can explore generative models and representation learning in a practical and engaging way.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Training](#training)
7. [Visualization](#visualization)
8. [Results](#results)
9. [Contributing](#contributing)
10. [License](#license)
11. [Contact](#contact)

---

## Introduction

Variational Autoencoders (VAEs) are powerful generative models that allow us to learn complex distributions from data. In this project, we use VAEs to learn a latent representation of fashion items from the FashionMNIST dataset. The goal is to visualize the 2D latent space, making it easier to understand how the model captures the relationships between different items.

The FashionMNIST dataset consists of 70,000 grayscale images of clothing items, such as shoes, shirts, and dresses. Each image is 28x28 pixels. This dataset serves as a great benchmark for testing generative models.

## Getting Started

To get started with this project, follow the steps outlined in the [Installation](#installation) section. After setting up the environment, you can begin training the model and visualizing the results.

You can also check the [Releases](https://github.com/Revanrie/Variational-Autoencoder-on-FashionMNIST/releases) section for pre-built binaries and model weights.

## Installation

To run this project, you will need Python 3.6 or higher. Follow these steps to set up your environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/Revanrie/Variational-Autoencoder-on-FashionMNIST.git
   cd Variational-Autoencoder-on-FashionMNIST
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset:
   The FashionMNIST dataset will be automatically downloaded when you run the training script.

## Usage

To train the Variational Autoencoder, run the following command:
```bash
python train.py
```

This will start the training process. You can monitor the training progress in the console.

After training, you can visualize the latent space using:
```bash
python visualize.py
```

This will generate plots that show how different items are represented in the latent space.

## Model Architecture

The architecture of the VAE consists of an encoder and a decoder:

- **Encoder**: This part compresses the input images into a lower-dimensional latent space. It consists of several convolutional layers followed by fully connected layers.

- **Decoder**: This part reconstructs the images from the latent space. It mirrors the encoder's architecture, using transposed convolutional layers.

The model uses the reparameterization trick to allow for backpropagation through the stochastic layer.

## Training

The training process involves minimizing the loss function, which consists of two parts:

1. **Reconstruction Loss**: This measures how well the model can reconstruct the input images.
2. **KL Divergence**: This measures how closely the learned latent distribution matches a standard normal distribution.

You can adjust the hyperparameters in `config.py` to fine-tune the model's performance.

## Visualization

Once the model is trained, you can visualize the latent space. The visualization script will create scatter plots showing how different classes of clothing items cluster in the latent space. This provides insights into how the model understands the relationships between different items.

## Results

After training, you can expect to see clear clusters in the latent space for different clothing categories. For example, shoes will be grouped together, while shirts will form another cluster. This shows that the model has learned meaningful representations of the data.

You can find sample results in the `results` folder after running the visualization script.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to create a pull request or open an issue.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please reach out via GitHub or open an issue in the repository.

Explore the [Releases](https://github.com/Revanrie/Variational-Autoencoder-on-FashionMNIST/releases) for downloadable content and pre-built models.

---

Thank you for checking out this repository! Enjoy exploring the world of Variational Autoencoders and the FashionMNIST dataset.