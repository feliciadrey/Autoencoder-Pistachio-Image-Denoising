# Autoencoder Image Denoising = Pistachio Dataset

Convolutional autoencoder to denoise pistachio images (100×100). Includes EDA, noise injection, baseline & modified autoencoders, and SSIM-based evaluation.


## Project Overview

This project builds convolutional autoencoders to **denoise images** of pistachios. Because the original dataset lacks noisy examples, synthetic Gaussian noise (mean=0.0, std=0.1) is added to create noisy inputs; the autoencoder learns to reconstruct clean images from noisy inputs.

Task objective: given a noisy 100×100 RGB image, reconstruct the clean image as closely as possible (measured with SSIM and MSE).


## Key Results

* **Dataset:** 1,074 RGB images resized to 100×100.
* **Train/Val/Test split:** 80% / 10% / 10% → `Train=859`, `Val=107`, `Test=108`.
* **Noisy vs Clean SSIM (before denoising):** average SSIM ≈ **0.22** (noisy images are substantially degraded).
* **Baseline autoencoder SSIM (test):** **\~0.9530**
* **Modified autoencoder SSIM (test):** **\~0.9575**

The modified (deeper + batch-norm) autoencoder achieves a small but consistent improvement in SSIM over the baseline.

---

## Dataset

* Provided via Google Drive link (downloaded with `gdown` in the notebook). The notebook expects a folder like `/content/dataset/A_23/*.jpg` after extraction.
* Images are read with OpenCV, converted BGR→RGB, resized to **100×100**, and normalized to `[0,1]`.
* Example shape used in the notebook: `(1074, 100, 100, 3)`.

---


## Exploratory Data Analysis (EDA) — summary

EDA steps performed in the notebook:

* Displayed sample images to inspect object location, cropping, and background consistency.
* Edge detection (Canny) applied to samples to observe contour clarity of pistachio objects.
* Plotted average per-channel color histograms to identify dominant pixel-value ranges.
* Computed per-channel mean and standard deviation.

**Key EDA takeaways:**

* Background is consistently dark which simplifies foreground extraction.
* Pistachio objects vary in pose and cropping, but contours are usually well-defined — promising for shape-preserving denoising.
* Pixel value histograms show brighter pixel modes for object regions and dark for backgrounds; normalization to `[0,1]` is appropriate.

---

## Preprocessing & Noise Generation

* Train/val/test split: `train_test_split` from scikit-learn with `random_state=123`.
* Gaussian noise injection function used: `mean=0.0`, `std=0.1` → noisy images clipped to `[0,1]`.
* SSIM between noisy and clean images was calculated to quantify degradation prior to training.

---

## Modeling

### Baseline autoencoder

* Input: `100×100×3` RGB image.
* Encoder: `Conv2D(32)->MaxPool`, `Conv2D(64)->MaxPool`, `Conv2D(64)`
* Decoder: `UpSampling->Conv2D(32)->UpSampling->Conv2D(3, activation='sigmoid')`
* Loss: MSE; Optimizer: `Adam()` (default LR)
* Trained: up to 100 epochs (notebook run) with validation monitoring.

### Modified autoencoder

* Deeper encoder with larger filters and BatchNormalization:

  * `Conv2D(64)->BatchNorm->MaxPool`
  * `Conv2D(128)->BatchNorm->MaxPool`
  * `Conv2D(256)` (latent)
* Decoder mirrors encoder with upsampling and progressively fewer filters: `128->64->decoded (3-channel sigmoid)`.
* Optimizer: `Adam(learning_rate=0.001)`; Loss: MSE
* Trained similarly (100 epochs in the notebook) and validated.

Design rationale: the modified model increases representational capacity and stabilizes training via batch normalization, yielding modest improvements in reconstruction quality.

---

## Evaluation

Metrics used:

* **MSE loss** during training/validation (used for optimization)
* **SSIM (Structural Similarity Index)** between original clean images and reconstructed outputs — reported on test set

Observed scores (notebook):

* Baseline SSIM ≈ **0.9530**
* Modified SSIM ≈ **0.9575**

**Interpretation:** both models produce high-fidelity reconstructions (SSIM > 0.95), with the modified autoencoder showing a small numeric improvement. Given the high SSIM for both, further improvements may require perceptual losses (e.g., VGG perceptual loss), adversarial training, or attention mechanisms if higher visual fidelity or sharper details are required.
