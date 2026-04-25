# 🧠 Representation Learning with Autoencoders (AE & VAE)

## 📌 Overview

This project explores unsupervised representation learning using:
- Autoencoder (AE)
- Variational Autoencoder (VAE)

The models are trained on the Medical MNIST dataset, which contains grayscale medical images from different anatomical regions such as AbdomenCT, ChestCT, Hand, HeadCT, and BreastMRI.

The goal is to understand how each model:
- Learns compact representations
- Reconstructs images
- Generates new samples (VAE)
- Handles noisy data

---

## 🎯 Objectives

- Implement an Autoencoder for reconstruction
- Implement a Variational Autoencoder with probabilistic latent space
- Visualize latent representations (2D & 3D)
- Compare AE vs VAE performance
- Generate new samples from latent space
- Explore denoising capability

---

## 🧱 Project Structure

medical-autoencoder/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── data_loader.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
│
└── notebook/
    └── experiment.ipynb

---

## ⚙️ Installation

Clone the repository:

git clone https://github.com/your-username/medical-autoencoder.git  
cd medical-autoencoder  

Install dependencies:

pip install -r requirements.txt

---

## 📂 Dataset

Download the dataset from:

https://www.kaggle.com/datasets/andrewmvd/medical-mnist

After downloading, extract it so the structure looks like:

data/
   AbdomenCT/
   ChestCT/
   Hand/
   HeadCT/
   BreastMRI/

---

## 🚀 Running the Project

To train the models:

python src/train.py

This will:
- Train AE and VAE for each class
- Show reconstruction results
- Plot training loss curves

---

## 🧠 Model Description

### Autoencoder (AE)
- Encoder: convolutional layers → latent vector
- Decoder: transposed convolutions
- Loss: reconstruction loss (MSE)

### Variational Autoencoder (VAE)
- Learns probabilistic latent space
- Uses reparameterization trick
- Loss:
  - Reconstruction loss
  - KL divergence

---

## 📊 Features

- Image reconstruction
- AE vs VAE comparison
- Latent space visualization (2D & 3D)
- Sample generation (VAE)
- Latent interpolation
- Denoising autoencoder
- Loss tracking (reconstruction + KL)
- tf.data pipeline

---

## 📈 Results

- AE produces sharper reconstructions
- VAE produces smoother images but better latent structure
- VAE allows generating new samples
- Latent space shows meaningful clustering
- Denoising improves robustness

---

## 📚 References

- TensorFlow Autoencoder Tutorial
- TensorFlow Variational Autoencoder Tutorial
- Medical MNIST Dataset (Kaggle)

---


## 🏁 Conclusion

- Autoencoder is best for reconstruction tasks
- VAE is best for generative modeling and structured representations

Both models highlight the trade-off between reconstruction quality and latent space regularization.
