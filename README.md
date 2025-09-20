# Pistachio Image Processing: Autoencoder & GAN

This repository applies **two deep learning techniques** on the same pistachio dataset (1,074 RGB images, resized to 100×100):

* **Autoencoder** → Image Denoising
* **Generative Adversarial Networks (GANs)** → Image Generation

Together, these projects illustrate complementary applications: **denoising** improves image fidelity, while **GANs** expand data diversity.

---

## Dataset

* **Source**: Pistachio image dataset provided via Google Drive link.
* **Format**: 1,074 RGB images → resized to **100×100**, normalized to `[0,1]`.
* **Splits**: Train 80% (859), Validation 10% (107), Test 10% (108).
* **Notes**: Dataset used consistently for both Autoencoder and GAN tasks.

---

## Part 1: Autoencoder for Image Denoising

### Overview

A convolutional autoencoder is trained to reconstruct clean pistachio images from artificially noised versions. Gaussian noise (`mean=0.0, std=0.1`) is added to simulate corruption.

**Task Objective:** Given a noisy input image, reconstruct the clean image.

### Exploratory Data Analysis (EDA)

* Sample visualization to inspect background consistency and object cropping.
* Edge detection (Canny) confirms well-defined pistachio contours.
* Color histograms show distinct modes for foreground vs background.
* Conclusion: Dataset is clean and well-suited for denoising tasks.

### Architectures

**Baseline Autoencoder**

* Encoder: `Conv2D(32)->MaxPool`, `Conv2D(64)->MaxPool`, `Conv2D(64)`
* Decoder: `UpSampling->Conv2D(32)->UpSampling->Conv2D(3, sigmoid)`
* Loss: MSE; Optimizer: Adam (default LR)

**Modified Autoencoder**

* Deeper encoder with BatchNorm layers
* Encoder: `Conv2D(64)->BatchNorm->MaxPool`, `Conv2D(128)->BatchNorm->MaxPool`, `Conv2D(256)`
* Decoder: mirrors encoder with upsampling + fewer filters
* Loss: MSE; Optimizer: Adam(lr=0.001)

### Key Results

* Noisy vs Clean SSIM (before denoising): **0.22**
* Baseline Autoencoder SSIM: **\~0.9530**
* Modified Autoencoder SSIM: **\~0.9575**

**Interpretation:** Both models achieve high-fidelity reconstructions (>0.95 SSIM). The modified version shows a modest improvement.

---

## Part 2: GAN for Pistachio Image Generation

### Overview

A Generative Adversarial Network (GAN) is trained to generate new pistachio images from random noise. Models are evaluated with the **Fréchet Inception Distance (FID)**.

**Task Objective:** Generate new, realistic pistachio images that match the real distribution.

### Architectures

**Baseline GAN**

* Generator: Dense → Reshape → Conv2D (64, 32, 16, ReLU) → Conv2D(3, tanh)
* Discriminator: Conv2D (16, 32, 64, ReLU) → Flatten → Dense (sigmoid)
* Optimizer: Adam(lr=1e-4)

**Modified GAN**

* Generator: Deeper filters (128→64→32) with BatchNorm + ReLU
* Discriminator: LeakyReLU, Dropout, `padding='same'`
* Optimizer: Adam(lr=2e-4, β1=0.5)

### Key Results

* Baseline GAN FID: **239.83**
* Modified GAN FID: **180.84**

**Visual Inspection:**

* Baseline: blurrier, less structured.
* Modified: sharper, smoother, more realistic.

---

## Evaluation Metrics

* **Autoencoder**: SSIM, MSE
* **GAN**: Fréchet Inception Distance (FID)

Both tasks are complementary:

* Autoencoder → Enhances input quality (denoising)
* GAN → Expands dataset with synthetic samples
